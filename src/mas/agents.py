"""Agents. LLM agents only *build chats* and *parse outputs*; the topology batches the calls."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import load_prompt
from .data import Number, extract_answer
from .faults import render_faulty_message
from .llm import Chat, Generation


@dataclass
class Message:
    sender: str
    content: str
    answer: Number | None
    faulty: bool = False                 # produced by the injected faulty agent
    accepted: bool = True                # verifier verdict (True if no verifier)
    verifier_output: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    meta: dict = field(default_factory=dict)


def format_peers(msgs: list[Message]) -> str:
    return "\n\n---\n\n".join(f"Solution from {m.sender}:\n{m.content}" for m in msgs)


def _msg(sender: str, gen: Generation) -> Message:
    return Message(sender, gen.text, extract_answer(gen.text),
                   prompt_tokens=gen.prompt_tokens,
                   completion_tokens=gen.completion_tokens, llm_calls=1)


class SolverAgent:
    is_llm = True

    def __init__(self, name: str, skeptical: bool = False):
        self.name, self.skeptical = name, skeptical

    def chat(self, item: dict, prior: list[Message]) -> Chat:
        system = load_prompt("solver_system")
        if self.skeptical:
            system += load_prompt("skeptical_addon")
        user = (load_prompt("solver_with_peer").format(question=item["question"],
                                                       peer_solutions=format_peers(prior))
                if prior else load_prompt("solver_alone").format(question=item["question"]))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, gen: Generation) -> Message:
        return _msg(self.name, gen)


class FaultyAgent:
    """Scripted: ignores its inputs and sends the pre-computed wrong solution."""
    is_llm = False

    def __init__(self, name: str, faults: dict, style: str):
        assert style in ("neutral", "persuasive")
        self.name, self.faults, self.style = name, faults, style

    def respond(self, item: dict) -> Message:
        f = self.faults[item["qid"]]
        return Message(self.name, render_faulty_message(f, self.style), f["wrong"],
                       faulty=True, meta={"style": self.style})


class AggregatorAgent:
    is_llm = True
    name = "aggregator"

    def __init__(self, skeptical: bool = False):
        self.skeptical = skeptical

    def chat(self, item: dict, msgs: list[Message]) -> Chat:
        system = load_prompt("aggregator_system")
        if self.skeptical:
            system += load_prompt("skeptical_addon")
        user = (load_prompt("aggregator_user").format(question=item["question"],
                                                      peer_solutions=format_peers(msgs))
                if msgs else load_prompt("aggregator_empty").format(question=item["question"]))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, gen: Generation) -> Message:
        return _msg(self.name, gen)


_VERDICT = re.compile(r"VERDICT\s*:\s*(ACCEPT|REJECT)", re.IGNORECASE)


class VerifierAgent:
    """Checks one solver message before it is forwarded. Unparseable verdict -> accept."""

    def chat(self, item: dict, msg: Message) -> Chat:
        return [{"role": "system", "content": load_prompt("verifier_system")},
                {"role": "user", "content": load_prompt("verifier_user").format(
                    question=item["question"], solution=msg.content)}]

    @staticmethod
    def apply(msg: Message, gen: Generation) -> None:
        found = _VERDICT.findall(gen.text)
        msg.accepted = not (found and found[-1].upper() == "REJECT")
        msg.verifier_output = gen.text
        msg.prompt_tokens += gen.prompt_tokens
        msg.completion_tokens += gen.completion_tokens
        msg.llm_calls += 1
