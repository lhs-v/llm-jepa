# 개입 판단 JEPA 실험 인수인계

## 현재 상태와 다음 실행

이 실행 경로는 **상황과 정책을 입력받아 개입 판단 JSON을 생성하는 모델**을 위한 실험 골격이다. 같은 데이터와 기본 모델에서 8개 학습 목적함수를 선택하고, 생성·JSON·판단 지표를 비교할 수 있다. 데이터 구조, 모델 로더, 채팅 템플릿, 손실, 실행 루프를 분리했다.

- 기본 모델: `google/gemma-4-E2B-it`.
- 기본 commit: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- 본 실험 설정: 단일 H100 80GB, BF16, 양자화 없음. **H100에서 실행하지 않았으므로 실제 메모리 사용량과 처리량은 미검증이다.**
- 로컬 4비트 실행은 코드와 메모리 경로의 호환성 확인용이다. BF16 본 실험의 결과로 취급하지 않는다.
- 예시 개입 데이터는 모두 합성 데이터다. 실제 운영 평가셋이나 효과 검증 자료가 아니다.
- 성능 향상은 아직 측정하지 않았다. 손실 감소나 짧은 학습 성공만으로 일반화·판단 품질 향상을 주장할 수 없다.
- 작업 브랜치: `feat/intervention-jepa-experiments`. 다른 머신에서는 사용자 fork의 이 브랜치를 선택한다. 기본 브랜치만 clone하면 이번 구현이 포함되지 않을 수 있다.

```bash
git clone --branch feat/intervention-jepa-experiments https://github.com/lhs-v/llm-jepa.git
cd llm-jepa
```

소스·설정·합성 예제·문서는 Git으로 전달한다. `.venv/`, `.cache/`, `outputs/`는 제외되므로 환경을 새로 설치하고 모델을 준비한다. 아래 검증 표의 `outputs/` 경로는 개발 머신의 로컬 기록이며, 어댑터·로그가 필요하면 별도로 전달하거나 기록된 명령으로 재생성한다.

기존 `finetune.py`, `finetune_gemma4.py`, `evaluate_gemma4.py`의 실행 경로와 구분해서 사용한다. 기존 Gemma 설정 설명은 [GEMMA4.md](GEMMA4.md), 새 데이터 계약은 [DATA_FORMAT.md](docs/DATA_FORMAT.md)에 있다.

## 1. 목적함수와 8개 recipe

기호는 다음과 같다.

- `X`: 정책과 상황만 포함한 입력. 기본 형식은 `Policy:\n{policy}\n\nSituation:\n{situation}`이다.
- `J`: 정답 JSON 객체를 일정한 키 순서와 공백 규칙으로 직렬화한 문자열.
- `R`: 데이터에 제공한 정답 설명. 직접 JSON 생성 입력에 넣지 않는다.
- `C_J`: `X → J`의 assistant 토큰 교차 엔트로피.
- `C_R`: 별도 설명 프롬프트의 `X → R` 교차 엔트로피.
- `C_RJ`: Gemma 사고 채널의 `R`을 생성한 뒤 `J`를 생성하는 순차 교차 엔트로피.
- `D_XJ`, `D_XR`: 독립적으로 인코딩한 두 표현의 `1 − cosine_similarity` 평균.
- `λ = experiment.jepa_weight`, `α = experiment.rationale_weight`, `β = experiment.sequential_weight`. 기본값은 각각 `0.1`, `0.5`, `1.0`이다.

| recipe ID | 학습 목적함수 | 기본 가중치에서 R 필요 | 우선 비교할 추론 모드 |
|---|---|---|---|
| `sft` | `C_J` | 아니요 | OFF |
| `json_jepa` | `C_J + λ D_XJ` | 아니요 | OFF |
| `rationale_jepa` | `C_J + λ D_XR` | 예 | OFF |
| `multitask` | `C_J + α C_R` | 예 | OFF |
| `multitask_jepa` | `C_J + α C_R + λ D_XR` | 예 | OFF |
| `cot` | `β C_RJ` | 예 | ON; OFF도 별도 비교 가능 |
| `cot_jepa` | `β C_RJ + λ D_XR` | 예 | ON; OFF도 별도 비교 가능 |
| `mixed_jepa` | `C_J + β C_RJ + λ D_XR` | 예 | OFF와 ON 각각 |

recipe와 평가 모드는 독립적인 설정이다. **`cot`를 선택해도 `evaluation.thinking`이 자동으로 ON이 되지 않는다.** Gemma 평가 시 `--thinking on` 또는 `off`를 명시해 결과를 구분한다. ON은 사고 채널 사용을 요청하는 프롬프트 설정이며, 실제 사고 생성 여부는 `generated_thought_fraction`으로 확인한다. OFF에서도 예기치 않은 사고 채널 출력이 생기면 원문과 발생 여부를 기록한다.

`cot`와 `cot_jepa`는 직접 OFF 응답 경로를 학습하지 않는다. 이 둘의 OFF 평가는 학습과 추론 조건이 달라지는 진단 실험이다. OFF 성능의 주 비교 대상은 `sft`, `rationale_jepa`, `multitask`, `multitask_jepa`, `mixed_jepa`로 둔다.

가중치가 0인 항은 실행하지 않는다. 필요한 R도 활성 손실에 따라 결정한다. 예를 들어 `rationale_jepa`에서 `λ=0`이면 직접 JSON 손실만 남고 R을 요구하지 않는다. 모든 생성 손실을 0으로 만들어 JEPA만 학습하는 설정은 허용하지 않는다.

CE는 branch마다 학습 대상 토큰 전체의 평균이다. `C_RJ`에서는 R, J, 사고 구분자와 종료 토큰을 함께 학습하며 R과 J에 별도의 구간 가중치를 주지 않는다. `β`는 순차 branch 전체의 가중치다.

### JEPA 구현에서 유지하는 조건

원래 LLM-JEPA의 `k=0` 설정을 유지한다. 예측기는 항등이며 추가 `[PRED]` 토큰, STP, EMA teacher, stop-gradient를 사용하지 않는다. 동일한 LoRA가 삽입된 디코더로 양쪽 view를 인코딩하고 **양쪽 모두에 기울기**를 전달한다.

`D_XR`의 두 view는 각각 `Enc(X)`와 `Enc(R)`이다. R view에 X나 J를 합치지 않으며, `X → J`의 프롬프트에도 R을 넣지 않는다. `multitask`의 설명 생성과 `cot`의 순차 생성은 별도 활성 branch에만 존재한다. 따라서 설명을 활용하는 학습과 추론 시 설명을 생성하는 동작을 나누어 비교할 수 있다.

Gemma는 각 독립 user 대화의 마지막 `<turn|>` 표현을 사용한다. 템플릿이 붙이는 뒤쪽 줄바꿈 토큰은 제외한다. 일반 채팅 포맷은 오른쪽 padding을 제외한 마지막 토큰을 사용한다. CE는 실제 학습 대상 토큰 수, JEPA는 예제 수로 정규화하므로 길이가 다른 microbatch와 마지막 불완전 누적 그룹을 올바르게 처리한다.

## 2. 실행 명령

아래 명령은 저장소 루트에서 가상환경을 활성화한 뒤 실행한다. 설치된 Python을 직접 지정하려면 Linux에서는 `.venv/bin/python`, Windows에서는 `.venv\Scripts\python.exe`를 사용한다. Python 3.12 기준으로 새 환경을 만들 때는 Linux에서 `bash setup_gemma4.sh`, Windows에서 `./setup_gemma4.ps1`을 실행한다. CUDA PyTorch를 먼저 설치한 기존 환경에는 아래 요구사항을 적용한다.

```bash
python -m pip install -r requirements-gemma4.txt
python train_experiment.py --config configs/intervention_h100.json --prepare-only
```

`--prepare-only`는 설정, 데이터, 토크나이저, 실제 채팅 템플릿, 활성 branch의 토큰 길이를 확인한다. 모델 가중치를 로드하거나 GPU 학습을 실행하지 않는다. 토크나이저와 설정이 캐시에 없으면 해당 파일은 다운로드할 수 있다.

이 단계는 결과를 표준 출력으로 보여주며 학습 run을 만들지 않는다. 제공된 파일은 다음과 같다.

| 파일 | 용도 |
|---|---|
| `configs/intervention_h100.json` | H100 BF16 본 실험의 시작 설정 |
| `configs/intervention_local_smoke.json` | 짧은 로컬 CUDA 4비트 호환성 확인 |
| `configs/intervention_nested_fields.json` | 중첩 필드 매핑 예시 |
| `examples/intervention_train.jsonl`, `examples/intervention_eval.jsonl` | 합성 개입 판단 예제 |
| `examples/intervention_nested.jsonl` | 중첩 record 형식의 합성 예제 |
| `examples/intervention_target.schema.json` | 제공 예제의 출력 schema |

짧은 H100 학습 확인 후 같은 설정으로 본 실험을 실행한다. 아래 단계의 실행 성공은 성능 검증과 별개다.

```bash
python train_experiment.py --config configs/intervention_h100.json --recipe rationale_jepa --output-dir outputs/intervention-rationale-smoke --max-steps 2 --max-samples 8
python train_experiment.py --config configs/intervention_h100.json --recipe rationale_jepa --output-dir outputs/intervention-rationale
python train_experiment.py --config configs/intervention_h100.json --recipe sft --output-dir outputs/intervention-sft
```

`--recipe`, `--output-dir`, `--max-steps`, `--max-samples`는 자주 바꾸는 설정의 단축 옵션이다. 그 밖의 설정은 `--set dotted.key=value`를 반복해서 지정한다. 값은 JSON으로 해석할 수 있고 문자열도 받는다.

```bash
python train_experiment.py --config configs/intervention_h100.json --recipe multitask_jepa --output-dir outputs/intervention-multitask-jepa --set experiment.jepa_weight=0.1 --set experiment.rationale_weight=0.5 --set training.seed=7
```

**경로 규칙:** JSON 설정 파일과 `--set`으로 지정한 상대 파일 경로는 설정 파일의 부모 디렉터리를 기준으로 해석한다. 예를 들어 `configs/intervention_h100.json`에서 저장소의 `datasets/`를 가리키려면 `../datasets/...`를 쓴다. `training.output_dir`, `model.cache_dir`, 명시적인 `./`·`../` 로컬 모델 경로에도 같은 규칙이 적용된다. Hub 모델 ID인 `google/gemma-4-E2B-it`는 로컬 경로로 변환하지 않는다.

단축 옵션인 학습 `--output-dir`은 현재 작업 디렉터리를 기준으로 절대 경로로 바꾼다. 따라서 위 명령의 `outputs/...`는 저장소 루트 아래 경로다. 반면 `--set training.output_dir=../outputs/...`는 설정 파일의 부모 디렉터리 기준이다. 단축 옵션과 같은 키의 `--set`을 함께 주면 단축 옵션이 우선한다.

학습은 내용이 있는 기존 출력 디렉터리를 거부한다. 비교 실험마다 새로운 출력 경로를 쓴다. 저장한 run을 평가할 때에는 다음과 같이 어댑터와 모델 metadata를 함께 읽는다.

```bash
python evaluate_experiment.py --run-dir outputs/intervention-rationale --output-dir outputs/intervention-rationale-eval --thinking off
python evaluate_experiment.py --run-dir outputs/intervention-rationale --output-dir outputs/intervention-rationale-eval-on --thinking on
```

기본 모델을 평가하려면 `--run-dir` 대신 `--config configs/intervention_h100.json`을 사용한다. 평가의 `--run-dir`과 `--output-dir`은 현재 작업 디렉터리를 기준으로 해석한다. 평가 출력은 이미 존재하는 디렉터리 자체를 거부하므로 빈 디렉터리도 미리 만들지 않는다.

저장 run 평가에서는 `data.*`, `evaluation.*`, `model.device`, `model.cache_dir`만 override할 수 있다. 모델 ID, revision, 프롬프트를 바꾸려면 기본 모델 평가인 `--config` 경로를 사용한다. run의 원래 모델과 프롬프트를 보존해야 어댑터 평가의 의미가 유지된다.

## 3. 산출물과 다시 불러오기

| 학습 run 경로 | 내용 |
|---|---|
| `adapter/` | 최종 LoRA 어댑터. 병합된 전체 모델이 아님 |
| `resolved_config.json` | 기본값과 override, 해석된 경로를 포함한 실행 설정 |
| `model_metadata.json` | 원래 모델 경로, 실제 Hub commit, tokenizer revision, dtype·장치·양자화·동결 파라미터 정보 |
| `dataset_manifest.json` | 데이터·스키마 식별 정보, 선택 예제, 길이 검증 및 제외 내역 |
| `runtime.json` | 실행 환경·라이브러리, 소스 파일 해시, 실제 설정 해시, optimizer 기본값 |
| `metrics.jsonl` | optimizer step 단위 손실·실행 기록 |
| `completed.json` | 정상 완료 기록 |
| `checkpoints/step-000001/` 등 | 선택적 중간 `adapter/`, `resolved_config.json`, `model_metadata.json` snapshot |

`training.save_every_steps`가 양수이면 중간 snapshot을 저장한다. **어댑터 snapshot은 정확한 학습 재개 기능이 아니다.** optimizer, scheduler, RNG, 데이터 위치를 포함한 전체 상태 복원을 구현하지 않았으므로 중단 전과 동일한 업데이트를 이어간다고 가정하지 않는다.

어댑터는 원래 기본 모델이 있어야 로드할 수 있다. Hub 모델은 설정을 먼저 읽어 commit을 확인하고, 같은 commit으로 기본 모델과 토크나이저를 고정한다. 로컬 모델 디렉터리는 Hub commit이 없어 `resolved_revision=null`이며, `local_source_sha256`에 모델·토크나이저 파일의 상대 경로와 해시를 기록한다. 저장 run 평가 전에 파일 목록·해시를 대조하고 변경·누락·추가를 거부한다. `.git`·`.cache` 같은 숨김 디렉터리는 해시에서 제외한다. 어댑터와 로그는 로컬 기본 모델 폴더 밖에 저장한다.

run의 `resolved_config.json`에는 당시 머신의 절대 경로가 남는다. 새 머신에서 학습할 때는 이 파일을 복사해 쓰기보다 `configs/`의 상대 경로 설정을 출발점으로 삼는다. 저장한 Hub 모델 어댑터를 다른 머신에서 평가할 때는 데이터·스키마·캐시 경로를 override한다. 로컬 기본 모델은 경로와 파일을 함께 보존해야 하며, 해시가 없는 이전 로컬 run은 검증 가능한 원본 snapshot으로 새로 만들어야 한다.

```bash
python evaluate_experiment.py --run-dir outputs/transferred-run --output-dir outputs/transferred-eval --thinking off --set data.eval_file=/srv/jepa/datasets/eval.jsonl --set data.target_schema=/srv/jepa/datasets/target.schema.json --set data.max_samples=null --set model.cache_dir=/srv/hf-cache
```

평가 출력에는 `predictions.jsonl`, `metrics.json`, `resolved_config.json`, `model_metadata.json`, `evaluation_manifest.json`이 생긴다. 예제별 생성 원문, 분리된 JSON 답과 사고 채널, 오류, 토큰 수, 지연 시간을 보존한다.

## 4. 평가 해석

생성은 `do_sample=False`로 실행한다. 원문을 special token을 보존한 상태로 디코딩한 뒤 Formatter가 Gemma 사고 채널과 답을 분리한다. R 정답은 평가 프롬프트에 넣지 않는다. 미완성 사고 채널과 잘못된 구분자는 format 오류로 남긴다. Markdown fence를 제거하거나 JSON 일부를 임의 추출하는 보정은 하지 않는다.

- JSON 문법 유효율, 선택한 JSON Schema 유효율, format 유효율을 따로 기록한다.
- `data.target_schema`는 자체 완결된 단일 schema 파일을 대상으로 한다. 내부 `$defs`와 `#/$defs/...` 참조는 사용할 수 있지만 다른 파일·네트워크의 `$ref`는 지원하지 않는다. 외부 정의는 파일 안으로 옮겨 사용한다.
- 구조적 exact match는 JSON 객체의 키 순서와 서식 공백에 영향받지 않는다. Boolean `true`와 숫자 `1`은 같다고 처리하지 않는다.
- 판단 필드는 `evaluation.decision_field`로 지정한다. 기본은 `intervene`이며 `decision.intervene`처럼 중첩 경로도 가능하다. 값은 Boolean이어야 한다.
- 혼동행렬과 일반 `accuracy`는 정답과 예측이 모두 Boolean인 예제만 분모에 포함한다. `strict_accuracy`는 Boolean 정답이 있는 모든 예제를 포함하여 잘못된 예측을 실패로 반영한다. `prediction_coverage`와 invalid 건수를 반드시 같이 본다.
- schema 통과 여부와 판단 필드의 Boolean 유효성은 별도 지표다. schema 위반이 있지만 판단 필드가 Boolean인 출력은 판단 지표에 포함될 수 있다.
- 분모가 0인 지표는 `null`이다. 실제 사고 생성률, 평균 출력 토큰 수, `model.generate` 구간의 지연 시간도 함께 비교한다.

같은 데이터 split, 기본 모델 commit, seed, LoRA 설정, 업데이트 수, 평가 모드를 맞춘다. JEPA와 추가 생성 branch는 계산량이 다르므로 총 학습 시간도 보고한다. H100 본 실험과 여러 seed의 결과가 확보되기 전에는 recipe 순위를 결론 내리지 않는다.

### 연구 가설의 한계

`D_XR` 감소 자체는 판단 성능이나 내부 추론 능력의 향상을 의미하지 않는다. 짧고 반복적인 R은 상황의 중요한 차이를 지우는 방향으로 정렬을 유도할 수 있다. 같은 정책에서 개입 정답이 반대인 사례, 정책만 바뀌면 판단이 달라지는 사례를 평가에 포함한다. R의 오류나 사후 정당화도 학습 신호가 되므로 근거 품질을 별도로 검토한다.

우선 `sft`와 `rationale_jepa`를 같은 조건에서 비교하고, `multitask`와 `multitask_jepa`로 설명 생성 지도와 임베딩 정렬의 기여를 구분한다. `λ=0` 및 여러 λ·seed 비교가 필요하다. 현재 코드는 표현 붕괴 측정, R 품질 검증, hyperparameter 탐색, 데이터 split 생성을 자동 수행하지 않는다.

## 5. 변경 지점

| 변경 목적 | 우선 수정할 파일 | 계약 |
|---|---|---|
| 설정 추가·기본값·검증 | `jepa_experiments/config.py` | JSON 설정만 지원. 알려지지 않은 키와 호환되지 않는 모드를 거부 |
| 데이터 형식·필드 매핑 | `jepa_experiments/data.py` | 외부 record를 `Example(id, situation, policy, rationale, target)`로 변환 |
| 모델 클래스·dtype·LoRA·디코더 | `jepa_experiments/models.py` | `load_model`은 `(model, tokenizer, metadata)`, `get_decoder`는 같은 LoRA를 사용하는 hidden-state 디코더 반환 |
| 프롬프트·마스킹·사고 구분자 | `jepa_experiments/formatting.py` | native template, 정확한 토큰 접두사, 독립 view, 생성 원문 파싱 |
| recipe·손실·누적 정규화 | `jepa_experiments/objectives.py` | 활성 branch만 만들고 direct 입력에 R이 들어가지 않도록 유지 |
| 학습 루프·저장·기록 | `jepa_experiments/training.py` | optimizer update 전 손실 성분별 backward, 기록과 어댑터 저장 |
| 생성·지표 | `jepa_experiments/evaluation.py` | 정답 R 없이 생성; format/JSON/schema/판단 실패를 분리 |
| CLI | `train_experiment.py`, `evaluate_experiment.py` | 설정·override 해석과 출력 디렉터리 관리 |

### 모델을 바꿀 때

`model_name_or_path`만 교체해 모든 모델을 지원한다고 약속하지 않는다. 같은 계열에서도 토크나이저, decoder 위치, LoRA 대상 모듈, context 길이와 메모리가 다를 수 있다.

1. Dense Gemma 4는 `backend=gemma4`의 명시적 텍스트 로더를 사용한다. 원래 멀티모달 checkpoint의 텍스트 경로를 매핑하고, 텍스트 가중치 누락 시 실패한다. MoE 및 Gemma4UV는 이 로더의 지원 대상이 아니다.
2. 다른 decoder-only 모델은 우선 `backend=auto_causal_lm`, `chat_format=standard`, `pooling=last_nonpad`로 연결한다. native chat template과 EOS/PAD가 필요하다. PAD가 없고 EOS가 있으면 EOS를 오른쪽 padding으로 사용하고 metadata에 남긴다.
3. `target_modules`를 해당 모델의 projection 이름에 맞춘다. `decoder_path`는 PEFT를 벗긴 기본 모델을 기준으로 한 점 경로이며, LM 출력층이 아닌 `last_hidden_state` 디코더여야 한다.
4. 일반 모델의 native template가 완성 대화와 생성 프롬프트의 정확한 토큰 접두사를 제공하는지 작은 실제 모델 테스트로 확인한다. 알 수 없는 사고 형식에는 별도 Formatter/모델 어댑터 구현이 필요하다. 현재 `standard`는 OFF만 지원하며 순차 recipe와 ON을 거부한다.
5. 새로운 모델은 작은 forward/backward, 동결 임베딩 dtype, 어댑터 저장·재로드, 독립 view, 출력 파싱을 먼저 확인한다. 모델별 전체 호환성을 추정하지 않는다.

기본 가중치를 동결하고 LoRA만 학습한다. Gemma의 큰 PLE 임베딩을 FP32로 바꾸는 범용 k-bit 전처리는 호출하지 않는다. gradient checkpointing은 `use_reentrant=False`이다. 4비트는 CUDA 전용이며 자동 CPU offload나 다중 GPU 실행은 제공하지 않는다.

## 6. 최종 검증 기록

검증일: 2026-09-15. 로컬 환경: Python 3.12.14, PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.20.0, bitsandbytes 0.50.2, jsonschema 4.26.0. 개발 중 개별 테스트 결과를 전체 검증 결과로 간주하지 않는다.

| 검증 | 최종 상태 | 증거·건수 |
|---|---|---|
| 기존 경로와 새 모듈 전체 테스트 | 통과 | `.venv/Scripts/python.exe -m pytest -q`: 126 passed, 실패·skip 없음. upstream Torch 비권장 경고와 기존 PEFT tiny fixture 경고 15건 |
| 8개 recipe prepare | 통과 | 캐시된 공식 tokenizer로 각 recipe 8개 예제의 전체 활성 branch 토큰화; 최대 202 tokens, 제외 없음. `outputs/intervention-prepare-matrix.json`; `--recipe <ID> --prepare-only`로 재현 |
| 의존성 검증 | 통과 | `.venv/Scripts/python.exe -m pip check`: No broken requirements found |
| 작은 실제 모델 학습·저장·평가 | 통과 | `tests/test_experiment_training.py`: 8개 recipe 각각 2 optimizer step, 불완전 누적 그룹, snapshot·최종 adapter 저장, 실제 평가 CLI와 재로드 확인 |
| 로컬 E2B `rationale_jepa` 학습 | 4비트 진단 실행 완료 | `outputs/intervention-rationale-jepa-smoke/`; optimizer 2 step, CE·JEPA 유한, gradient norm 양수, peak allocated 6.859 GiB |
| 위 로컬 어댑터 재로드·OFF 평가 | 실행 완료; 엄격한 JSON 평가 실패 | `outputs/intervention-rationale-jepa-smoke-eval/`; 4건 모두 Markdown fence 출력, JSON·schema 유효 0/4, 사고 채널 0/4 |
| raw JSON 지시를 추가한 E2B 재점검 | 학습·저장·재로드·OFF 평가 통과 | `outputs/intervention-rationale-jepa-raw-json-smoke/` 및 `...-eval/`; 2 step, peak allocated 6.924 GiB; 합성 4건 JSON·schema 유효 4/4, 판단 일치 4/4, 사고 채널 0/4, 전체 JSON exact match 2/4 |
| H100 BF16 본 실험 | 미실행 | 메모리·처리량·품질 결과 없음 |
| 실제 개입 판단 평가 및 여러 seed 비교 | 미실행 | 성능 향상 측정 없음 |

다음 담당자는 실제 정책과 대상 분포를 반영하는 학습·평가 데이터를 준비하고, 장면·템플릿·정책 단위로 split을 분리한 뒤 H100에서 짧은 실행과 본 비교 실험을 순서대로 수행한다. 같은 run을 덮어쓰지 말고 설정과 산출물을 함께 보관한다.

위 로컬 메모리 수치는 해당 4비트·짧은 입력·작은 배치 실행에서 PyTorch가 할당한 GPU 메모리의 최대값이다. H100 BF16 본 실험의 메모리 요구량이나 처리량 추정치로 사용하지 않는다.

첫 로컬 평가에서는 `strict_accuracy=0`, `prediction_coverage=0`이었고 유효 Boolean 예측 쌍이 없어 일반 `accuracy=null`이었다. 모델이 출력한 Markdown fence를 제거하지 않았기 때문이다. 이 실패의 원문과 설정을 보존했다.

현재 예제 설정에는 `Output raw JSON with no Markdown fences or surrounding text. Begin with { and end with }.` 지시를 추가했다. 같은 seed·학습 데이터에서 새 어댑터를 2 step 학습한 재점검은 위 합성 4건에서 코드 블록 없이 JSON을 생성했다. 긍정 예제 두 건의 instruction 문구가 정답과 달라 전체 JSON exact match는 2/4였다. 프롬프트도 바뀌었고 SFT 대조군도 없는 작은 진단이므로 JEPA의 성능 향상 근거로 사용할 수 없다.

재점검 명령은 다음과 같다. 이미 사용한 디렉터리를 재사용하지 않도록 실행할 때 출력 이름을 바꾼다.

```bash
python train_experiment.py --config configs/intervention_local_smoke.json --output-dir outputs/intervention-rationale-jepa-raw-json-smoke
python evaluate_experiment.py --run-dir outputs/intervention-rationale-jepa-raw-json-smoke --output-dir outputs/intervention-rationale-jepa-raw-json-smoke-eval --set evaluation.max_new_tokens=128
```
