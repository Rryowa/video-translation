from openai import OpenAI

def translate_segments(segments, model_name="Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", thinking_budget=500):
    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local")
    
    for seg in segments:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Translate the following video subtitle text to Russian. Output ONLY the translation. No quotes, no explanation."},
                {"role": "user", "content": seg['text']}
            ],
            temperature=0.3,
            extra_body={"thinking_budget_tokens": thinking_budget}
        )
        seg['text'] = response.choices[0].message.content.strip()
    return segments
