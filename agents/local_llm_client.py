import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMClient:
    def __init__(self, model_name=None, max_new_tokens=512):
        self.model_name = model_name or os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
        self.max_new_tokens = max_new_tokens

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto",
        )

    def chat(self, messages, temperature=0.2, max_tokens=None):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded