from openai import OpenAI

def translate_segments(segments, language="Russian", thinking_budget=500):
    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local")
    
    for seg in segments:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": f"Translate the following video subtitle text to {language}. Output ONLY the translation. No quotes, no explanation."},
                {"role": "user", "content": seg['text']}
            ],
            temperature=0.3,
            extra_body={"thinking_budget_tokens": thinking_budget}
        )
        seg['text'] = response.choices[0].message.content.strip()
    return segments
