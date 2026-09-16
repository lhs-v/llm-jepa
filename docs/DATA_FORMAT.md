# 개입 판단 데이터 형식

학습과 평가는 UTF-8 JSONL을 읽는다. **한 줄에 JSON 객체 하나**를 넣는다. 빈 줄은 무시하며 UTF-8 BOM은 허용한다. 이 문서의 내용과 저장소의 개입 예시 데이터는 모두 합성 예시다.

외부 형식은 최종적으로 다음 객체로 변환된다.

```text
Example(id: str, situation: str, policy: str, rationale: str | None, target: dict)
```

`situation`과 `policy`는 비어 있지 않은 문자열이어야 한다. `target`은 JSON 객체이며 문자열로 저장한 JSON 객체도 받을 수 있다. 배열, 숫자, Boolean 단독 값은 target이 될 수 없다. NaN과 Infinity 같은 비표준 수치는 거부한다.

실행 가능한 저장소 예시는 `examples/intervention_train.jsonl`, `examples/intervention_eval.jsonl`과 `examples/intervention_target.schema.json`이다. 중첩 예시는 `examples/intervention_nested.jsonl` 및 `configs/intervention_nested_fields.json`에 있다. 아래 문서 예시는 형식 설명을 위한 별도 schema를 사용하므로, 저장소 예제 schema를 그대로 적용하려면 그 파일의 필수 필드와 허용 값을 확인한다.

## 1. 기본 records 형식

```json
{"id":"synthetic-001","situation":"회의실에서 참석자가 화면 공유를 시작하지 못해 도움을 요청했다.","policy":"명시적인 도움 요청이 있을 때만 개입한다.","rationale":"도움을 명시적으로 요청했으므로 정책의 개입 조건에 해당한다.","target":{"intervene":true,"action":"offer_help"}}
{"id":"synthetic-002","situation":"회의실에서 참석자가 조용히 자료를 읽고 있다.","policy":"명시적인 도움 요청이 있을 때만 개입한다.","rationale":"도움 요청이 없으므로 개입 조건을 충족하지 않는다.","target":{"intervene":false,"action":"wait"}}
```

해당 데이터에 맞는 설정 부분은 다음과 같다. 아래는 `configs/`에 둔 JSON 설정을 가정한 경로 예시다.

```json
{
  "data": {
    "format": "records",
    "train_file": "../datasets/my_train.jsonl",
    "eval_file": "../datasets/my_eval.jsonl",
    "fields": {
      "id": "id",
      "situation": "situation",
      "policy": "policy",
      "rationale": "rationale",
      "target": "target"
    },
    "max_length": 512,
    "overlength": "error"
  }
}
```

`id`가 누락되거나 `data.fields.id=null`이면 원본 파일의 실제 줄 번호를 문자열 ID로 사용한다. ID를 제공할 때는 비어 있지 않은 문자열이어야 한다. 같은 파일에서 중복 ID가 있으면 오류이며, 일부 예제만 선택하는 실행에서도 파일 전체의 중복을 검사한다.

## 2. 중첩 필드와 공통 정책

점으로 구분한 경로로 중첩 객체의 필드를 선택할 수 있다. 배열 인덱스 선택과 JSONPath는 지원하지 않는다.

```json
{"key":"synthetic-nested-001","input":{"scene":"사용자가 도움을 요청했다."},"annotation":{"explanation":"요청이 있으므로 개입한다.","output":{"decision":{"intervene":true}}}}
```

```json
{
  "data": {
    "format": "records",
    "fields": {
      "id": "key",
      "situation": "input.scene",
      "policy": null,
      "rationale": "annotation.explanation",
      "target": "annotation.output"
    },
    "default_policy": "명시적인 도움 요청이 있을 때만 개입한다."
  },
  "evaluation": {
    "decision_field": "decision.intervene"
  }
}
```

policy 필드가 없거나 `null`이면 `data.default_policy`를 사용한다. 둘 다 없으면 오류다. record의 policy가 빈 문자열이면 fallback으로 대체하지 않고 오류로 처리한다. 설명이 필요하지 않은 recipe에서는 `data.fields.rationale=null`로 두어 읽지 않을 수 있다.

## 3. messages 형식

기존 한 턴 대화 데이터는 `data.format="messages"`로 읽는다. 허용 역할 순서는 `[user, assistant]` 또는 `[system, user, assistant]`이다. 여러 대화 턴, tool 메시지, 이미지·오디오 content 구조는 지원하지 않는다.

```json
{"id":"synthetic-message-001","messages":[{"role":"system","content":"명시적인 도움 요청이 있을 때만 개입한다."},{"role":"user","content":"사용자가 화면 공유를 도와 달라고 요청했다."},{"role":"assistant","reasoning":"명시적인 도움 요청이 있으므로 개입한다.","content":"{\"intervene\":true,\"action\":\"offer_help\"}"}]}
```

변환 규칙은 다음과 같다.

| 원본 | Example 필드 |
|---|---|
| 선택적 `system.content` | `policy` |
| `user.content` | `situation` |
| `assistant.reasoning` | `rationale` |
| `reasoning`이 없거나 null일 때 `assistant.reasoning_content` | `rationale` 대체 값 |
| `assistant.content` | `target`; JSON 객체 또는 JSON 객체 문자열 |
| `data.fields.id`가 가리키는 record 필드 | `id` |

system 메시지가 없으면 `data.default_policy`가 필요하다. 원본 system은 정책 데이터로 해석하며 학습 시 그대로 최상위 system 역할로 복사하지 않는다. 학습 프롬프트의 system은 `prompts.system` 또는 `prompts.rationale_system`에서 오고, 정책은 상황과 함께 X에 들어간다. messages 형식에서 situation·policy·rationale·target의 사용자 지정 필드 매핑은 사용하지 않는다.

## 4. J 정규화와 입력 분리

`target` 문자열은 먼저 JSON 객체로 파싱한다. 이후 J는 다음 규칙으로 생성한다.

```python
json.dumps(target, ensure_ascii=False, sort_keys=True,
           separators=(",", ":"), allow_nan=False)
```

예를 들어 `{"intervene": true, "action": "offer_help"}`는 `{"action":"offer_help","intervene":true}`가 된다. 한국어는 유니코드 이스케이프 없이 유지한다. 원본 JSON의 키 순서나 들여쓰기가 학습 정답에 영향을 주지 않는다.

`prompts.input_template`에는 `{policy}`와 `{situation}`을 모두 포함해야 한다. 그 외 치환 필드는 허용하지 않는다. 기본 입력은 다음과 같다.

```text
Policy:
{policy}

Situation:
{situation}
```

R과 J는 직접 JSON 생성의 입력 X에 포함되지 않는다. rationale JEPA의 target view는 R만 독립 user 대화로 인코딩한다. `multitask`의 R 생성이나 `cot`의 사고 채널 생성은 선택한 recipe의 별도 학습 branch다. 순차 학습용 R을 데이터에 적었다고 모든 추론에서 R을 입력으로 요구하는 것은 아니다.

## 5. R이 필요한 경우

| recipe | 기본 가중치에서 rationale |
|---|---|
| `sft`, `json_jepa` | 없어도 됨 |
| `rationale_jepa`, `multitask`, `multitask_jepa` | 필요 |
| `cot`, `cot_jepa`, `mixed_jepa` | 필요 |

실제 요구 여부는 활성 손실에서 계산한다. `rationale_weight=0`이면 설명 생성 branch를 만들지 않고, `jepa_weight=0`이면 JEPA view를 만들지 않는다. R을 사용하는 항이 모두 비활성일 때만 R을 생략할 수 있다. 모든 생성 손실을 비활성화한 실행은 허용하지 않는다.

**평가 데이터는 recipe와 관계없이 R이 없어도 된다.** 평가에는 X와 정답 J가 필요하다. `evaluation.thinking=on`이어도 정답 R을 프롬프트에 넣지 않는다. R 필드를 제공했다면 빈 문자열은 허용하지 않으므로, 평가 설명을 생략하려면 필드를 없애거나 null로 둔다.

## 6. JSON Schema와 판단 필드

출력 구조는 고정되어 있지 않다. `data.target_schema`로 별도 JSON Schema 파일을 지정하면 모든 학습·평가 target과 모델 예측을 그 schema로 검사한다. 예를 들어 다음 schema는 이 문서의 기본 records 예시에 맞는다.

스키마를 프롬프트에 자동 삽입하지 않는다. 출력 필드나 의미를 바꿀 때는 `prompts.system`의 출력 지시와 `evaluation.decision_field`도 함께 갱신한다. 상황이나 정책이 원본 데이터에 정답·근거를 이미 포함하는 경우까지 자동 검출하지는 않으므로 원본 데이터 단계에서 누출을 확인한다.

현재 지원 범위는 **자체 완결된 단일 schema 파일**이다. 같은 파일의 `$defs`와 `#/$defs/...` 같은 내부 `$ref`는 지원한다. 형제 파일이나 네트워크 주소를 가리키는 외부 `$ref`는 지원하지 않으므로 정의를 같은 파일에 넣는다. 여러 파일의 schema와 외부 reference resolver가 필요한 작업은 별도 확장 대상이다.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "intervene": {"type": "boolean"},
    "action": {"type": "string", "enum": ["offer_help", "wait"]}
  },
  "required": ["intervene", "action"],
  "additionalProperties": false
}
```

```json
{
  "data": {"target_schema": "../datasets/my_schema.json"},
  "evaluation": {"decision_field": "intervene"}
}
```

schema를 생략해도 학습 target은 JSON 객체여야 하며 평가의 기본 schema도 객체를 요구한다. `decision_field`는 Boolean 개입 판단 지표를 계산할 필드의 점 경로다. `"true"`, `1`, `null`, 누락 필드는 Boolean으로 변환하지 않는다. Boolean 판단 필드가 없는 작업도 JSON/schema/exact-match 지표는 얻을 수 있지만 해당 판단 지표의 분모가 없으면 `null`이 된다.

이 예시 schema는 개입 여부와 action의 논리적 일치까지 강제하지 않는다. 실제 정책의 출력 계약에 조건부 제약이 있다면 JSON Schema나 별도의 평가 지표로 명시한다.

## 7. 길이, 선택과 오류 처리

- 잘못된 JSON, 빈 필수 필드, 중복 ID, schema 위반은 파일 경로와 줄 번호를 포함한 오류로 처리한다. 잘못된 record를 조용히 버리지 않는다.
- `data.max_samples`는 파일 순서에서 앞의 N개를 선택한다. 선택 한도 뒤에 있는 record도 전체 유효성 검사를 수행한다. 무작위 표본 추출 옵션이 아니다.
- 학습에서는 선택 recipe의 모든 활성 CE branch와 독립 JEPA view를 실제 토크나이저로 인코딩한다. 어느 하나라도 `data.max_length`를 넘으면 기본 `overlength=error`에서 실패한다.
- `overlength=skip`이면 해당 예제 전체를 제외하고 ID·길이 내역을 기록한다. 상황, R, J를 중간에서 자르지 않는다. 모든 예제가 제외되면 학습 준비에 실패한다.
- 평가에서는 생성 프롬프트 길이에 같은 error/skip 정책을 적용한다. 생성 길이는 별도의 `evaluation.max_new_tokens`로 제한한다. 프롬프트 제한이 prompt+생성 전체의 context 적합성을 대신 검증하지는 않는다.
- `evaluation.max_samples`는 평가 생성에 적용하는 별도 한도다. `data.max_samples`도 지정되어 있으면 데이터 선택 한도 안에서 평가한다.

준비 단계에서 데이터 원문과 선택 결과, schema, 길이 검증 내역을 확인한다. 제외율이 높으면 길이나 데이터를 조정하고, 평가 분포가 달라지지 않았는지 점검한다.

## 8. split과 인수인계

같은 장면의 단어만 바꾼 예제, 같은 생성 템플릿의 변형, 같은 정책의 거의 동일한 설명이 train과 eval에 섞이면 일반화 성능을 과대평가할 수 있다. **장면·원본 템플릿·정책 묶음 단위로 split을 나누고** 근접 중복을 확인한다. 현재 loader는 파일을 읽을 뿐 split 생성이나 파일 간 중복 제거를 자동 수행하지 않는다.

데이터를 전달할 때 원본 생성 방식, 정책 버전, 라벨 정의, split 기준, 실제/합성 여부를 함께 기록한다. 파일·schema hash와 선택 ID는 실행 manifest에 남기고, 설정 JSON과 출력 run을 함께 보관한다. 실제 평가셋은 recipe와 가중치를 선택한 뒤 최종 비교에 사용한다.
