import logging
from openai import OpenAI

logger = logging.getLogger("translator")

def translate_segments(segments, language="Russian", thinking_budget=500):
    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local")
    logger.debug(f"Translating {len(segments)} segments to {language} via API at {client.base_url}")
    
    for i, seg in enumerate(segments, 1):
        original_text = seg['text']
        logger.debug(f"Segment {i}/{len(segments)}: Original: {original_text!r}")
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": f"Translate the following video subtitle text to {language}. Output ONLY the translation. No quotes, no explanation."},
                {"role": "user", "content": original_text}
            ],
            temperature=0.3,
            extra_body={"thinking_budget_tokens": thinking_budget}
        )
        translated_text = response.choices[0].message.content.strip()
        logger.debug(f"Segment {i}/{len(segments)}: Translated: {translated_text!r}")
        seg['text'] = translated_text
    return segments
