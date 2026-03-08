import openai
from config.settings import settings

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3-4b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "stepfun/step-3.5-flash:free",
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
