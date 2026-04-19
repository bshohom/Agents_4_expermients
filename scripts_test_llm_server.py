from agents.llm_client import LocalLLMClient

client = LocalLLMClient()

resp = client.chat(
    messages=[
        {
            "role": "system",
            "content": "You are a concise scientific planning assistant. Return only the final answer.",
        },
        {
            "role": "user",
            "content": "Propose one simple weld bead experiment varying travel speed.",
        },
    ],
    max_tokens=300,
    temperature=0.2,
)

print(resp)