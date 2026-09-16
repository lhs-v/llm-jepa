"""Extract policy, situation, and JSON answer from an existing SFT record."""

import json


def parse_jepa_record(record):
    """Read messages without re-rendering or modifying the original record."""
    messages = record["messages"]
    roles = [message["role"] for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError("Expected system, user, assistant in this order.")
    system_text, user_text, answer_text = [message["content"] for message in messages]
    if not all(isinstance(value, str) for value in (system_text, user_text, answer_text)):
        raise ValueError("Message content must be a string.")
    if not system_text.strip() or not user_text.strip():
        raise ValueError("System and user text must not be empty.")
    answer = json.loads(answer_text)
    if not isinstance(answer, dict):
        raise ValueError("Assistant content must be a JSON object string.")
    # json.loads accepts NaN/Infinity; reject them even inside the answer string.
    json.dumps(answer, allow_nan=False)
    return {"system": system_text, "user": user_text, "answer": answer}
