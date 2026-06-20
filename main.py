import argparse
import os
import torch
import time
import logging
from src.audio_extractor import extract_audio
from src.transcriber import transcribe_audio, free_model
from src.srt_generator import group_alignments, generate_srt
from src.video_burner import burn_subtitles
from src.translator import translate_segments

logger = logging.getLogger("main")

# Check for CUDA availability
if not torch.cuda.is_available():
    print("\n" + "="*80)
    print("WARNING: CUDA is not available to PyTorch. The models will run on CPU,")
    print("which is extremely slow and will consume a massive amount of system RAM (up to 32GB).")
    print("To install PyTorch with CUDA support (compatible with CUDA 12 and 13+):")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
    print("="*80 + "\n")

def process_video(video_path: str, language: str = "Russian", thinking_budget: int = 500):
    start_time = time.time()
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}.wav"
    srt_path = f"{base_name}.srt"
    output_video = f"{base_name}_subtitled.mp4"
    
    logger.info(f"Starting video processing for {video_path}")
    
    step_start = time.time()
    logger.info(f"Extracting audio to {audio_path}...")
    extract_audio(video_path, audio_path)
    logger.debug(f"Audio extraction completed in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info("Transcribing and aligning audio with Qwen3-ASR...")
    transcription = transcribe_audio(audio_path)
    logger.debug(f"ASR Transcription completed in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info("Grouping alignments into segments...")
    if transcription.time_stamps is None:
         raise ValueError("ASR transcription did not return timestamps. Ensure the forced aligner is properly loaded.")
    
    segments = group_alignments(transcription.time_stamps)
    logger.debug(f"Grouped into {len(segments)} segments in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info("Releasing ASR model resources...")
    # Free ASR model VRAM before translation
    free_model()
    logger.debug(f"Model released in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Translating segments to {language}...")
    segments = translate_segments(segments, language, thinking_budget)
    logger.debug(f"Translation completed in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Generating SRT file at {srt_path}...")
    generate_srt(segments, srt_path)
    logger.debug(f"SRT generated in {time.time() - step_start:.2f}s")
    
    step_start = time.time()
    logger.info(f"Burning subtitles into video: {output_video}...")
    burn_subtitles(video_path, srt_path, output_video)
    logger.debug(f"Subtitle burn completed in {time.time() - step_start:.2f}s")
    
    logger.info(f"Completed! Output video: {output_video} (Total time: {time.time() - start_time:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add subtitles to a video using Qwen3-ASR and FFmpeg.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--language", default="Russian", help="Target language for translation (default: Russian)")
    parser.add_argument("--thinking-budget", type=int, default=500, help="Thinking budget tokens for llama.cpp")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        
    process_video(args.video, args.language, args.thinking_budget)
