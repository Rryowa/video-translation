import argparse
import os
import logging
import time

logger = logging.getLogger("main")

def process_video(video_path: str, language: str, model_name: str, custom_style: str = None, max_words: int = 12, max_chars: int = 40, end_padding: float = 0.8):
    logger.info("Loading AI libraries (PyTorch, Transformers)... Please wait.")
    import torch
    from src.audio_extractor import extract_audio
    from src.transcriber import transcribe_audio, free_model
    from src.srt_generator import group_alignments, generate_srt
    from src.video_burner import burn_subtitles
    from src.translator import translate_segments

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("WARNING: CUDA is not available to PyTorch. The models will run on CPU,")
        print("which is extremely slow and will consume a massive amount of system RAM (up to 32GB).")
        print("Please see the PyTorch Installation Guide in README.md to install the correct CUDA version for your GPU.")
        print("="*80 + "\n")
    start_time = time.time()
    
    video_dir = os.path.dirname(os.path.abspath(video_path))
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    
    processed_dir = os.path.join(video_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    audio_path = os.path.join(processed_dir, f"{base_name}.wav")
    srt_path = os.path.join(processed_dir, f"{base_name}.srt")
    output_video = os.path.join(processed_dir, f"{base_name}_subtitled.mp4")
    
    if os.path.exists(output_video):
        logger.info(f"Skipping {video_path}: Output {output_video} already exists.")
        return
        
    logger.info(f"Starting video processing for {video_path}")
    
    # Check if model exists in Ollama (fail fast)
    import urllib.request, json, urllib.error
    logger.info(f"Checking if model '{model_name}' exists in Ollama...")
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            available_models = [m.get("name") for m in data.get("models", [])]
            # Match exact or with :latest
            if model_name not in available_models and f"{model_name}:latest" not in available_models:
                raise ValueError(f"Model '{model_name}' not found in Ollama. Available: {', '.join(available_models)}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Could not reach Ollama API at http://127.0.0.1:11434. Is Ollama running? Error: {e}")
    
    step_start = time.time()
    logger.info(f"Extracting audio to {audio_path}...")
    extract_audio(video_path, audio_path)
    logger.debug(f"Audio extraction completed in {time.time() - step_start:.2f}s")
    
    # Unload Ollama model to free VRAM for ASR
    logger.info(f"Unloading Ollama model '{model_name}' to free VRAM...")
    import urllib.request, json
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        logger.debug(f"Failed to unload Ollama model (normal if not running): {e}")

    step_start = time.time()
    logger.info("Transcribing and aligning audio with Qwen3-ASR...")
    transcription = transcribe_audio(audio_path)
    logger.debug(f"ASR Transcription completed in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info("Grouping alignments into segments...")
    if transcription.time_stamps is None:
        if not getattr(transcription, "text", "").strip():
            logger.warning("No speech detected in the audio. Creating empty subtitle file.")
            with open(srt_path, "w", encoding="utf-8") as f:
                pass
            logger.info(f"Completed! Output video remains unchanged or can be copied. Skipping burn.")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(srt_path):
                os.remove(srt_path)
            return
        else:
            raise ValueError("ASR transcription did not return timestamps despite detecting text. Ensure the forced aligner is properly loaded.")
    
    segments = group_alignments(transcription.time_stamps, max_words=max_words, max_chars=max_chars, end_padding=end_padding)
    logger.debug(f"Grouped into {len(segments)} segments in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info("Releasing ASR model resources...")
    # Free ASR model VRAM before translation
    free_model()
    logger.debug(f"Model released in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Translating segments to {language}...")
    segments = translate_segments(segments, language, model_name, custom_style)
    logger.debug(f"Translation completed in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Generating SRT file at {srt_path}...")
    generate_srt(segments, srt_path)
    logger.debug(f"SRT generated in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Burning subtitles into video: {output_video}...")
    burn_subtitles(video_path, srt_path, output_video)
    logger.debug(f"Subtitle burn completed in {time.time() - step_start:.2f}s")
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(srt_path):
        os.remove(srt_path)
        
    logger.info(f"Completed! Output video: {output_video} (Total time: {time.time() - start_time:.2f}s)")

if __name__ == "__main__":
    import importlib.util
    import sys
    if importlib.util.find_spec("torch") is None:
        print("\nERROR: PyTorch is not installed.")
        print("Please follow the Installation Guide in README.md to install the correct PyTorch version for your GPU.\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Add subtitles to a video using Qwen3-ASR and FFmpeg.")
    parser.add_argument("video", help="Path to input video file or folder")
    parser.add_argument("--language", default="Russian", help="Target language for translation (default: Russian)")
    parser.add_argument("--model", required=True, help="LLM model name for translation")
    parser.add_argument("--style", help="Path to a text file containing custom style instructions to append to the translation prompt.")
    parser.add_argument("--max-words", type=int, default=12, help="Maximum words per subtitle segment (default: 12)")
    parser.add_argument("--max-chars", type=int, default=40, help="Maximum characters per subtitle segment (default: 40)")
    parser.add_argument("--end-padding", type=float, default=0.8, help="Seconds to keep subtitles on screen after speech ends (default: 0.8)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    custom_style = None
    if args.style:
        if os.path.exists(args.style):
            with open(args.style, "r", encoding="utf-8") as f:
                custom_style = f.read().strip()
            logger.info(f"Loaded custom style from {args.style}")
        else:
            logger.warning(f"Custom style file '{args.style}' not found. Falling back to default system prompt.")
            
    video_input = args.video
    if os.path.isdir(video_input):
        valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        for file in os.listdir(video_input):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                file_path = os.path.join(video_input, file)
                process_video(file_path, args.language, args.model, custom_style, args.max_words, args.max_chars, args.end_padding)
    else:
        process_video(video_input, args.language, args.model, custom_style, args.max_words, args.max_chars, args.end_padding)
