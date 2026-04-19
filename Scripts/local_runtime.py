
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B")
DEFAULT_LOCAL_EMBED_MODEL = os.environ.get(
    "LOCAL_EMBED_MODEL",
    "/trace/group/tmousavi/gyunghuy/cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
)
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


_cached_llms: dict[str, "LocalQwenLLM"] = {}
_configured_embed_model_name: str | None = None

def get_embed_model_name() -> str:
    return os.environ.get("LOCAL_EMBED_MODEL", DEFAULT_EMBED_MODEL)

def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


class LocalQwenLLM:
    def __init__(self, model_name: str = DEFAULT_LOCAL_LLM_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, max_new_tokens: int = 550) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return strip_think_blocks(text)


def get_local_llm(model_name: str | None = None) -> LocalQwenLLM:
    resolved_model_name = model_name or os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL)
    if resolved_model_name not in _cached_llms:
        _cached_llms[resolved_model_name] = LocalQwenLLM(resolved_model_name)
    return _cached_llms[resolved_model_name]


def configure_embedding_model(model_name: str | None = None) -> None:
    global _configured_embed_model_name

    resolved_model_name = model_name or os.environ.get("LOCAL_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL)
    if _configured_embed_model_name == resolved_model_name:
        return

    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=get_embed_model_name(),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    _configured_embed_model_name = resolved_model_name
