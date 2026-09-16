# Gemma 4에서 기존 LLM-JEPA 실험하기

`finetune_gemma4.py`는 Gemma 4의 텍스트 디코더에 LoRA를 적용하는 실행 경로입니다. 본 실험은 **H100 80GB 한 장**을 가정하며, 기본 모델은 `google/gemma-4-E2B-it`입니다. 기본값은 **BF16, 양자화 없음, 길이 512, 배치 4, 기울기 누적 4, LoRA rank 64**입니다.

현재 H100에 접근할 수 없어 이 설정의 실제 메모리 사용량과 실행 가능 여부는 확인하지 않았습니다. 4비트 양자화는 작은 로컬 GPU에서 실행 경로를 확인할 때 선택할 수 있습니다.

## 방법과 적용 범위

목적함수는 원래 LLM-JEPA의 **정답 토큰 교차 엔트로피 + λ × Text/Code 표현의 코사인 거리**입니다. Text와 Code를 각각 독립적인 대화로 인코딩하고 양쪽 표현 모두에 기울기를 전달합니다. STP는 사용하지 않습니다.

- `k=0`: 원래 LLM-JEPA에서 허용하는 항등 예측기 설정으로, 추가 `[PRED]` 토큰이 없습니다. 토큰을 추가하거나 임베딩을 확장하지 않습니다.
- JEPA의 두 입력은 원본 Gemma 경로처럼 각각 `user` 역할로 구성합니다. 각 대화의 `<turn|>` 토큰 표현을 사용하며 뒤의 줄바꿈은 제외합니다.
- 정답 학습은 Gemma 4의 기본 대화 템플릿과 system 역할을 사용합니다. 생각하기 모드를 끄고 assistant의 답과 `<turn|>`만 학습합니다.
- 기본 모델의 가중치는 BF16으로 동결하고 LoRA만 학습합니다. 비재진입 방식의 gradient checkpointing을 사용합니다. 선택적으로 4비트를 사용할 때에도 큰 PLE 임베딩은 BF16으로 유지하며, 이를 FP32로 바꾸는 범용 k-bit 전처리를 호출하지 않습니다.
- 텍스트 전용 클래스를 명시하고 원본 체크포인트의 텍스트 가중치 경로를 변환합니다. 텍스트 가중치가 누락되면 실행을 중단합니다. 비전·오디오 가중치의 `UNEXPECTED` 로딩 보고는 텍스트 추출 과정에서 예상되는 출력입니다.

이는 원래 목적함수의 `k=0` 설정을 Gemma 4에 적용한 실행 경로입니다. 원본 `finetune.py`와 대화 템플릿 및 종료 토큰 학습 방식까지 동일한 재현은 아닙니다. 현재는 단일 GPU와 최종 어댑터 저장을 지원하며, 중간 체크포인트 재개·다중 GPU·`k>0` 예측 토큰은 구현하지 않았습니다.

E2B를 우선 대상으로 검증합니다. E4B·31B 같은 dense `gemma4` 모델은 모델 경로를 바꿔 시도할 수 있으나 메모리 요구량과 실행 여부를 별도로 확인해야 합니다. `gemma4uv` 구조의 12B 및 26B MoE에 대한 지원은 제공하지 않습니다. 이 설정의 성능 향상은 아직 측정하지 않았습니다.

E4B를 시도하려면 `--model_name_or_path google/gemma-4-E4B-it`로 바꾸고 아래 예제의 E2B 전용 `--revision` 인자는 제거하세요. 재현 실험에서는 해당 E4B 모델의 commit을 `--revision`으로 고정하세요.

## 1. 환경 설치

Linux 서버, NVIDIA GPU, Python 3.12를 기준으로 합니다. 저장소 디렉터리에서 실행합니다.

```bash
bash setup_gemma4.sh
```

Python 실행 파일을 지정할 수도 있습니다.

```bash
bash setup_gemma4.sh /path/to/python3.12
```

Windows에서는 PowerShell용 설치 스크립트를 사용합니다.

```powershell
.\setup_gemma4.ps1
```

Python이 PATH에 없으면 실행 파일을 지정합니다.

```powershell
.\setup_gemma4.ps1 -PythonExe 'C:\path\to\python.exe'
```

두 스크립트는 저장소의 `.venv`를 만들거나 재사용하고 PyTorch 2.11.0 CUDA 12.8과 버전을 고정한 의존성을 설치합니다. CUDA 12.8을 지원하는 NVIDIA 드라이버가 필요합니다. 모델 다운로드와 학습은 아래 명령으로 별도 실행합니다. 첫 모델 실행에는 Hugging Face 다운로드를 위한 네트워크와 디스크 공간이 필요합니다. 기본 캐시 위치는 `.cache/huggingface`입니다.

이하 본 실험 명령은 Linux Bash용입니다. Windows에서는 Python 경로를 `.\.venv\Scripts\python.exe`로 바꾸고 줄 연결 문자 `\`를 PowerShell의 백틱으로 바꾸세요.

## 2. 데이터 준비 확인

JSONL 한 줄에 `messages`를 넣습니다. 문자열 내용의 선택적 system 메시지와 user/assistant 한 쌍을 지원합니다.

```json
{"messages":[{"role":"system","content":"Convert natural language to regular expression."},{"role":"user","content":"a digit"},{"role":"assistant","content":"[0-9]"}]}
```

토크나이저만 내려받아 입력을 검사합니다. 모델 가중치는 로드하지 않습니다.

```bash
.venv/bin/python finetune_gemma4.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --train_file datasets/synth_train.jsonl \
  --output_dir outputs/gemma4-e2b-prepare \
  --max_length 512 --max_train_samples 8 --prepare_only
```

최대 길이를 넘는 예제는 중간을 자르지 않고 통째로 제외합니다. 출력된 사용·제외 건수를 확인하세요. 길이가 긴 데이터에서 제외 비율이 높다면 `--max_length`를 늘려야 하며, GPU 메모리 사용량도 증가합니다.

## 3. H100에서 2단계 학습 확인

```bash
.venv/bin/python finetune_gemma4.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --train_file datasets/synth_train.jsonl \
  --output_dir outputs/gemma4-e2b-h100-smoke \
  --quantization none --max_length 512 --batch_size 4 \
  --gradient_accumulation_steps 1 --lora_rank 64 --lbd 0.1 \
  --max_train_samples 8 --max_steps 2 --seed 42
```

이 실행은 다운로드, 모델 로딩, 손실 계산, 역전파, 어댑터 저장을 확인하는 용도입니다. 두 번의 업데이트로 성능을 평가할 수는 없습니다.

## 4. LLM-JEPA와 일반 LoRA 비교

```bash
.venv/bin/python finetune_gemma4.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --train_file datasets/synth_train.jsonl \
  --output_dir outputs/gemma4-e2b-jepa \
  --quantization none --max_length 512 --batch_size 4 \
  --gradient_accumulation_steps 4 --lora_rank 64 --lbd 0.1 \
  --num_epochs 4 --seed 42
```

일반 LoRA는 같은 명령에 `--regular`를 추가하고 `--output_dir outputs/gemma4-e2b-lora`로 바꿉니다. 이 경우 JEPA용 추가 인코딩을 생략하고 정답 토큰 손실만 학습합니다.

데이터, 모델 revision, 시드, LoRA rank, 업데이트 수를 맞춰 비교하세요. JEPA는 추가 인코딩 때문에 같은 업데이트 수에서도 계산 비용이 더 큽니다. 성능과 학습 시간을 함께 기록하고, 결론을 내리기 전 여러 시드로 반복하는 것이 좋습니다. 학습 결과는 병합된 전체 모델 대신 LoRA 어댑터로 저장됩니다.

## 5. 저장한 어댑터로 생성·평가

```bash
.venv/bin/python evaluate_gemma4.py \
  --adapter-dir outputs/gemma4-e2b-jepa \
  --input-file datasets/synth_test.jsonl \
  --output-dir outputs/gemma4-e2b-jepa-eval \
  --max-examples 100 --max-new-tokens 128
```

평가기는 학습 결과의 `run_config.json`에서 기본 모델, revision, 양자화 설정을 읽고 어댑터를 다시 로드합니다. 평가 출력 디렉터리는 새 경로여야 합니다. `generations.jsonl`에는 예측과 정답이, `metrics.json`에는 문자열 exact match 집계가 저장됩니다. 일반 LoRA도 어댑터 경로와 출력 디렉터리를 바꿔 같은 테스트 예제로 평가합니다.

단일 입력을 확인하려면 `--input-file` 대신 `--prompt 'a digit' --system-prompt 'Convert natural language to regular expression.'`를 사용합니다.

이 스크립트는 생성 문자열의 exact match를 보고합니다. 정규식 실행 결과의 의미적 동등성이나 태스크 일반화 성능은 별도 평가가 필요합니다.

## 6. 선택 사항: 작은 로컬 GPU에서 4비트 실행 확인

RTX 5070 12GB에서 짧게 확인할 때 사용할 PowerShell 명령입니다. 본 실험은 위의 H100 BF16 설정으로 진행합니다.

```powershell
.\.venv\Scripts\python.exe finetune_gemma4.py `
  --model_name_or_path google/gemma-4-E2B-it `
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 `
  --train_file datasets/synth_train.jsonl `
  --output_dir outputs/gemma4-e2b-local-smoke `
  --quantization 4bit --max_length 128 --batch_size 1 `
  --gradient_accumulation_steps 1 --lora_rank 32 --lbd 0.1 `
  --max_train_samples 8 --max_steps 2 --seed 42
```

## 7. 검증 상태

2026-09-15, Windows / RTX 5070 / Python 3.12.14에서 확인했습니다.

- 고정 버전 라이브러리 설치 및 `pip check` 통과.
- 테스트 13개 통과: 공식 토크나이저의 학습 마스킹, 독립적인 두 JEPA 경로의 역전파, PLE·KV 공유와 체크포인팅, 누적 배치 기울기, 멀티모달 체크포인트에서 텍스트 가중치의 정확한 복원, BF16 GPU 업데이트, 어댑터 재로딩 등을 포함합니다.
- 실제 E2B 가중치로 4비트 LoRA JEPA **2 optimizer steps**, 유한한 CE·JEPA 손실과 어댑터 저장 확인. 해당 짧은 실행의 PyTorch 최대 할당 메모리는 약 **6.97 GiB**였습니다. H100 BF16 설정의 메모리 추정치가 아닙니다.
- 저장한 어댑터를 다시 로드해 SYNTH 테스트 입력 2개에서 생성 완료. 64토큰 제한의 동작 확인이므로 벤치마크 결과나 성능 향상 근거로 사용하지 않습니다.
- Linux 설치 스크립트의 Bash 구문 검사와 PowerShell 설치 스크립트의 구문 검사 통과.

**H100에서 실제 BF16 E2B 학습과 전체 데이터의 성능 비교는 아직 실행하지 않았습니다.** 서버에서는 위의 2스텝 확인 명령부터 실행하세요.

현재 로컬 검증 산출물은 `outputs/gemma4-e2b-compat-smoke/`와 `outputs/gemma4-e2b-compat-eval/`에 있습니다. 환경·모델 캐시·검증 산출물은 Git에서 제외됩니다.

테스트를 다시 실행하려면 `.venv/bin/python -m pytest tests -q`를 사용하세요. 공식 토크나이저 검증은 캐시가 없으면, BF16 GPU 검증은 CUDA가 없으면 건너뜁니다.

## 출처

- [공식 LLM-JEPA 저장소](https://github.com/galilai-group/llm-jepa)
- [Transformers 5.17.0 Gemma 4 문서](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/gemma4)
- [Google Gemma 4 E2B 모델 카드](https://huggingface.co/google/gemma-4-E2B-it)
