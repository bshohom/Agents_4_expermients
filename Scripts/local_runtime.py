from __future__ import annotations

import os
from pathlib import Path

import torch
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B")
DEFAULT_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

_LLM_CACHE: dict[str, "LocalQwenLLM"] = {}
_EMBED_MODEL_CONFIGURED: str | None = None


class LocalQwenLLM:
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
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

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 550,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if temperature is not None and temperature > 0:
            generate_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            generate_kwargs.update(
                do_sample=False,
            )

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text


def get_shared_llm(model_name: str | None = None) -> LocalQwenLLM:
    resolved_model = model_name or os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LLM_MODEL)
    if resolved_model not in _LLM_CACHE:
        _LLM_CACHE[resolved_model] = LocalQwenLLM(resolved_model)
    return _LLM_CACHE[resolved_model]


def configure_embedding_model(embed_model: str | None = None) -> None:
    global _EMBED_MODEL_CONFIGURED

    resolved_embed_model = embed_model or os.environ.get("LOCAL_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if _EMBED_MODEL_CONFIGURED == resolved_embed_model:
        return

    model_name_or_path = str(Path(resolved_embed_model)) if "/" in resolved_embed_model or resolved_embed_model.startswith(".") else resolved_embed_model

    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    _EMBED_MODEL_CONFIGURED = resolved_embed_model
