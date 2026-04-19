"""Prompt-driven experiment proposer.

This agent:
1. receives a BOM dict and retrieved paper-card context
2. loads a prompt template from disk
3. fills the template with serialized context
4. calls the local LLM via LocalLLMClient

The prompt file should use these placeholders:
- {bom_json}
- {retrieved_context}
- {user_goal}
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.llm_client import LocalLLMClient


class ProposerAgent:
    def __init__(
        self,
        prompt_file: str | Path,
        llm_client: LocalLLMClient | None = None,
    ) -> None:
        self.prompt_file = Path(prompt_file)
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        bom: dict,
        user_goal: str,
        retrieved_cards: list[dict],
    ) -> str:
        system_prompt = self.prompt_file.read_text(encoding="utf-8")

        user_payload = {
            "user_goal": user_goal,
            "bom": bom,
            "retrieved_cards": retrieved_cards,
        }

        messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\n\nReturn only the final proposed experiment. Do not include thinking process or chain-of-thought.",
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2),
            },
        ]

        return self.llm.chat(messages=messages, max_tokens=700, temperature=0.2)