# 기존 SFT 환경으로 옮기는 JEPA 데이터 준비 코드

이 도구는 reasoning 유무가 다른 두 JSONL을 `id`로 연결하고, 각 레코드에
JEPA 학습용 `jepa_context`와 `jepa_rationale` 문자열을 추가합니다.
Python 표준 라이브러리만 사용하며 GPU나 DeepSpeed가 필요하지 않습니다.

## 옮길 파일

기존 프로젝트 루트 기준으로 다음 파일을 복사합니다.

```text
scripts/prepare_jepa_data.py
src/data/jepa_data.py
```

저장소에는 `scripts/__init__.py`, `src/__init__.py`, `src/data/__init__.py`도
포함되어 있습니다. 새 프로젝트에서는 함께 복사하고, 이미 같은 파일이 있는
프로젝트에서는 기존 파일을 유지하세요.

## 입력 형식

두 파일 모두 각 줄에 아래 구조의 JSON 객체가 있어야 합니다.

```json
{
  "id": "sample-001",
  "messages": [
    {"role": "system", "content": "상황을 분석하고 개입 여부를 JSON으로 답하세요."},
    {"role": "user", "content": "등록된 사용자가 도어락 인증에 성공했습니다."},
    {"role": "assistant", "content": "{\"is_proactive_action_required\": false, \"reasoning\": \"정상적인 출입으로 이상 신호가 없습니다.\"}"}
  ],
  "text": "기존 학습에 사용하던 전체 대화 문자열",
  "language": "ko",
  "category": "home"
}
```

- `id`는 비어 있지 않은 문자열 또는 정수이고, 각 파일 안에서 유일해야 합니다.
  두 파일의 ID 집합은 같아야 합니다. 정수 `1`과 문자열 `"1"`은 다른 ID입니다.
- `messages`는 정확히 `system → user → assistant` 순서이고, 내용은 문자열입니다.
  Assistant 내용은 JSON 객체 문자열이어야 합니다. JSON 앞뒤의 별도 설명이나
  코드 블록, thinking 채널은 이 파서에서 지원하지 않습니다.
- reason 파일의 답변에는 비어 있지 않은 `reasoning` 문자열이 필요합니다.
  no_reason 파일에서는 `reasoning`을 생략하거나 `null`, `""`로 두세요.
- 같은 ID의 user 내용은 앞뒤 공백을 제외하고 같아야 합니다. System 내용은 각
  실험의 출력 지침에 따라 달라도 됩니다. 정책의 의미가 같은지는 직접 확인하세요.
- `is_proactive_action_required`는 JSON boolean이어야 합니다. `reasoning`을
  제외한 답변은 두 파일에서 같아야 합니다. 개입이 `false`인 경우에만 `intents`
  생략과 빈 배열을 비교 과정에서 동일하게 취급합니다. 원본 답변은 수정하지 않습니다.

## 실행

프로젝트 루트에서 실행합니다. 출력 폴더는 아직 존재하지 않는 경로를 지정하세요.

```bash
uv run --active python -m scripts.prepare_jepa_data \
  --reason data/train_reason.jsonl \
  --no-reason data/train_no_reason.jsonl \
  --output-dir data/jepa_ready
```

PowerShell에서는 위 명령을 한 줄로 입력하면 됩니다. `uv`를 사용하지 않는 환경에서는
`uv run --active python` 대신 해당 환경의 `python`을 사용하세요.

출력은 `data/jepa_ready/reason.jsonl`, `data/jepa_ready/no_reason.jsonl`입니다.
각 출력은 해당 입력 파일의 순서를 유지합니다. 원본 파일과 원래 필드 값은 유지하며
다음 두 필드만 추가합니다.

| 추가 필드 | 값 |
| --- | --- |
| `jepa_context` | 해당 파일의 `system`과 `user`를 `Policy:\n...\n\nSituation:\n...`으로 연결 |
| `jepa_rationale` | 같은 ID의 reason 파일 답변에서 추출한 `reasoning` |

`jepa_context`에는 assistant 답변을 넣지 않습니다. no_reason 출력의 기존 assistant
답변에도 reasoning을 추가하지 않습니다. 두 파일을 메모리에 읽어 검증한 뒤 출력하며,
기존 출력 폴더나 이미 JEPA 필드가 있는 레코드는 덮어쓰지 않습니다.

## 학습 코드 연결은 다음 단계

이 도구는 데이터 준비 단계입니다. 기존 SFT 코드가 사용하는 `text`와 `messages`는
다시 만들지 않으며, 두 표현의 내용이 일치하는지도 검사하지 않습니다.

Dataset 전처리와 collator에서 추가 필드를 토큰화하고, Trainer에서 상황/정책과
rationale의 표현으로 JEPA loss를 계산하도록 연결해야 합니다. 연결 전까지는
`jepa.enabled: false`를 유지하세요. reason 실험과 no_reason 실험 모두 각 파일의
기존 assistant 답변을 SFT 정답으로 사용합니다.

이 출력은 기존 사내 SFT 환경에 옮기는 용도입니다. 저장소의 `train_experiment.py`는
[별도 데이터 형식](DATA_FORMAT.md)을 사용하므로 이 파일을 바로 입력으로 받도록
연결되어 있지는 않습니다.

## 검증

```bash
python -m pytest tests/test_prepare_jepa_data.py -q
```
