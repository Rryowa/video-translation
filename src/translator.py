import logging
import json
import urllib.request

logger = logging.getLogger("translator")

def translate_segments(segments, language, model_name, custom_style=None):
    api_url = "http://127.0.0.1:11434/api/chat"
    logger.debug(f"Translating {len(segments)} segments to {language} via API at {api_url}")
    
    system_prompt = (
        f"You are a professional video subtitle translator. Translate the input text to {language}.\n\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the direct translation. Do NOT add conversational filler, preambles, explanations, or quotes.\n"
        "- The translated text must be extremely concise and must NOT contain twice or more words than the original text.\n"
        "- Maintain the original tone and pacing. You can add 1-3 words on 1 phrase based on your personality.\n"
        "- CRITICAL: Since this is for subtitles, keep it short.\n"
    )
    if custom_style:
        system_prompt += f"\n\nAdditional style instructions:\n{custom_style.replace('{language}', language)}"
    
    for i, seg in enumerate(segments, 1):
        original_text = seg['text']
        logger.debug(f"Segment {i}/{len(segments)}: Original: {original_text!r}")
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_text}
            ],
            "think": False,
            "stream": False
        }
        
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                translated_text = response_data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Translation failed for segment {i}: {e}")
            translated_text = original_text # fallback
            
        logger.debug(f"Segment {i}/{len(segments)}: Translated: {translated_text!r}")
        seg['text'] = translated_text
        
    return segments
