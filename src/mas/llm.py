"""Batched chat-generation wrapper with token accounting.

Every agent turn is batched across all questions of a condition, so one `generate`
call = one agent position for N questions. This is what makes the grid feasible on a T4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Chat = list[dict]  # [{"role": "system"|"user"|"assistant", "content": str}, ...]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLM:
    def __init__(self, name: str, backend: str = "vllm", dtype: str = "float16",
                 max_new_tokens: int = 512, temperature: float = 0.0, seed: int = 0,
                 gpu_memory_utilization: float = 0.85):
        self.name, self.backend = name, backend
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        if backend == "vllm":
            from vllm import LLM as VLLM, SamplingParams
            self._llm = VLLM(model=name, dtype=dtype, seed=seed,
                             gpu_memory_utilization=gpu_memory_utilization)
            self._params = SamplingParams(temperature=temperature,
                                          max_tokens=max_new_tokens, seed=seed)
            self.tokenizer = self._llm.get_tokenizer()
        elif backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.manual_seed(seed)
            self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=getattr(torch, dtype), device_map="auto").eval()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def _render(self, chat: Chat) -> str:
        return self.tokenizer.apply_chat_template(chat, tokenize=False,
                                                  add_generation_prompt=True)

    def generate(self, chats: list[Chat], batch_size: int = 16) -> list[Generation]:
        if not chats:
            return []
        prompts = [self._render(c) for c in chats]
        if self.backend == "vllm":
            outs = self._llm.generate(prompts, self._params, use_tqdm=False)
            return [Generation(o.outputs[0].text, len(o.prompt_token_ids),
                               len(o.outputs[0].token_ids)) for o in outs]
        return self._generate_hf(prompts, batch_size)

    def _generate_hf(self, prompts: list[str], batch_size: int) -> list[Generation]:
        import torch
        tok, res = self.tokenizer, []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j in range(len(batch)):
                ids = new[j][new[j] != tok.pad_token_id]
                res.append(Generation(tok.decode(ids, skip_special_tokens=True),
                                      int(enc["attention_mask"][j].sum()), int(len(ids))))
        return res


class MockLLM:
    """For tests / dry runs: maps a chat to a response string with a Python function."""

    def __init__(self, fn: Callable[[Chat], str]):
        self.fn = fn
        self.name = "mock"

    def generate(self, chats: list[Chat], **_) -> list[Generation]:
        return [Generation(self.fn(c), 0, 0) for c in chats]
