import openai
from config.settings import settings

client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=settings.NVIDIA_API_KEY,
)

MODELS = [
    "mistralai/mistral-small-4-119b-2603",
    "google/gemma-3-27b-it",
    "meta/llama-3.3-70b-instruct",
]

PROMPT = "What was the significance of the Battle of Stalingrad in World War II? Answer in 2-3 sentences."

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"max_tokens=3000")
    print("-" * 60)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable historian assistant."},
                {"role": "user", "content": PROMPT},
            ],
            max_tokens=3000,
            temperature=0.0,
        )
        content = r.choices[0].message.content
        if content:
            print(f"OK ({len(content)} chars): {content[:200]}")
        else:
            print(f"EMPTY RESPONSE (content is None/empty)")
        print(f"Finish reason: {r.choices[0].finish_reason}")
    except Exception as e:
        print(f"ERROR: {e}")
