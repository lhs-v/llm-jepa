"""Native chat templates, exact assistant masks, and explicit thought channels."""
from string import Formatter as StringFormatter


DEFAULT_PROMPTS = {
    "system": "Return only the target JSON object. Do not include explanations.",
    "rationale_system": "Explain how the policy applies to the situation.",
    "input_template": "Policy:\n{policy}\n\nSituation:\n{situation}",
}


class Formatter:
    def __init__(self, tokenizer, chat_format="gemma4", prompts=None, pooling="turn_end"):
        if chat_format not in ("gemma4", "standard"):
            raise ValueError("chat_format must be gemma4 or standard")
        if pooling not in ("turn_end", "last_nonpad"):
            raise ValueError("pooling must be turn_end or last_nonpad")
        if not tokenizer.chat_template:
            raise ValueError("A tokenizer with a native chat template is required")
        if chat_format == "standard" and pooling != "last_nonpad":
            raise ValueError("Standard chat templates require last_nonpad pooling")
        self.tokenizer, self.chat_format, self.pooling = tokenizer, chat_format, pooling
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}
        template = self.prompts["input_template"]
        if not isinstance(template, str) or not template.strip():
            raise ValueError("input_template must be a nonempty string")
        for _, field, _, _ in StringFormatter().parse(template):
            if field is not None and field not in {"situation", "policy"}:
                raise ValueError("input_template may format only situation and policy")
        self.turn_end_id = None
        if chat_format == "gemma4":
            if "<turn|>" not in tokenizer.get_vocab():
                raise ValueError("Gemma 4 formatting requires the <turn|> tokenizer")
            self.turn_end_id = tokenizer.convert_tokens_to_ids("<turn|>")

    def input_text(self, example):
        return self.prompts["input_template"].format(situation=example.situation, policy=example.policy)

    def _prompt_messages(self, example, rationale_task=False):
        system = self.prompts["rationale_system" if rationale_task else "system"]
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": self.input_text(example)})
        return messages

    def messages(self, example, branch="direct"):
        if branch not in ("direct", "rationale", "sequential"):
            raise ValueError("branch must be direct, rationale, or sequential")
        if branch == "sequential" and self.chat_format != "gemma4":
            raise ValueError("Sequential thought-channel supervision is supported only for Gemma 4")
        result = self._prompt_messages(example, rationale_task=branch == "rationale")
        answer = example.rationale if branch == "rationale" else example.target_text
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Supervised answer must be nonempty text")
        assistant = {"role": "assistant", "content": answer}
        if branch == "sequential":
            if not isinstance(example.rationale, str) or not example.rationale.strip():
                raise ValueError("Sequential rationale must be nonempty text")
            assistant["reasoning"] = example.rationale
        result.append(assistant)
        return result

    def _template(self, messages, *, tokenize, generation=False, thinking=False):
        if thinking and self.chat_format != "gemma4":
            raise ValueError("Thinking generation is supported only for Gemma 4")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=generation,
            enable_thinking=thinking, **({"return_dict": False} if tokenize else {}),
        )

    def render_prompt(self, example, thinking=False, rationale_task=False):
        return self._template(self._prompt_messages(example, rationale_task),
                              tokenize=False, generation=True, thinking=thinking)

    def _through_turn_end(self, ids):
        try:
            end = len(ids) - 1 - ids[::-1].index(self.turn_end_id)
        except ValueError as error:
            raise ValueError("Gemma 4 chat template omitted its final turn marker") from error
        return ids[:end + 1]

    def encode_supervision(self, example, branch):
        messages = self.messages(example, branch)
        thinking = branch == "sequential"
        full = list(self._template(messages, tokenize=True, thinking=thinking))
        if self.chat_format == "gemma4":
            full = self._through_turn_end(full)
            if full.count(self.tokenizer.bos_token_id) != 1:
                raise ValueError("Gemma 4 conversation must contain exactly one BOS token")
        prompt_text = self._template(messages[:-1], tokenize=False, generation=True, thinking=thinking)
        prefix = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("Chat template prompt must be an exact token prefix with a nonempty answer")
        return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]}

    def encode_view(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Independent view must be nonempty text")
        ids = list(self._template([{"role": "user", "content": text}], tokenize=True))
        return self._through_turn_end(ids) if self.pooling == "turn_end" else ids

    def _strip_endings(self, text):
        endings = [self.tokenizer.eos_token]
        if self.chat_format == "gemma4":
            endings.append(self.tokenizer.convert_ids_to_tokens(self.turn_end_id))
        text = text.strip()
        while True:
            ending = next((item for item in endings if item and text.endswith(item)), None)
            if ending is None:
                return text
            text = text[:-len(ending)].rstrip()

    def parse_generation(self, raw):
        """Keep JSON untouched; report malformed/unfinished Gemma channels separately."""
        text = self._strip_endings(raw)
        result = {"answer": text, "rationale": None, "thought_generated": False, "format_error": None}
        if self.chat_format != "gemma4":
            return result
        start, end = "<|channel>thought", "<channel|>"
        result["thought_generated"] = start in text
        if text.startswith(start):
            body = text[len(start):]
            if not body.startswith("\n"):
                result["format_error"] = "Thought channel header must end with a newline"
            if end not in body:
                result.update(answer="", rationale=body.strip(), format_error="Unfinished thought channel")
                return result
            rationale, answer = body.split(end, 1)
            result.update(answer=answer.strip(), rationale=rationale.strip())
            if "<|channel>" in rationale or "<|channel>" in answer or end in answer:
                result["format_error"] = "Multiple or nested thought channels"
        elif "<|channel>" in text or end in text or "<|think|>" in text:
            result["format_error"] = "Unexpected Gemma channel delimiter"
            if text.startswith(end):
                result["answer"] = text[len(end):].strip()
        return result
