import argparse
import os
from src.audio_extractor import extract_audio
from src.transcriber import transcribe_audio, free_model
from src.srt_generator import group_alignments, generate_srt
from src.video_burner import burn_subtitles
from src.translator import translate_segments

def process_video(video_path: str, do_translate: bool = False, model_name: str = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", thinking_budget: int = 500):
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}.wav"
    srt_path = f"{base_name}.srt"
    output_video = f"{base_name}_subtitled.mp4"
    
    print(f"Extracting audio to {audio_path}...")
    extract_audio(video_path, audio_path)
    
    print("Transcribing and aligning audio with Qwen3-ASR...")
    transcription = transcribe_audio(audio_path)
    
    print("Grouping alignments into segments...")
    if transcription.time_stamps is None:
        raise ValueError("ASR transcription did not return timestamps. Ensure the forced aligner is properly loaded.")
    
    segments = group_alignments(transcription.time_stamps)
    
    # Free ASR model VRAM before translation
    free_model()
    
    if do_translate:
        print("Translating segments to Russian...")
        segments = translate_segments(segments, model_name, thinking_budget)
    
    print(f"Generating SRT file at {srt_path}...")
    generate_srt(segments, srt_path)
    
    print(f"Burning subtitles into video: {output_video}...")
    burn_subtitles(video_path, srt_path, output_video)
    
    print(f"Completed! Output video: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add subtitles to a video using Qwen3-ASR and FFmpeg.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--translate", action="store_true", help="Translate subtitles to Russian using local LLM")
    parser.add_argument("--translation-model", default="Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", help="Model name for translation API")
    parser.add_argument("--thinking-budget", type=int, default=500, help="Thinking budget tokens for llama.cpp")
    args = parser.parse_args()
    process_video(args.video, args.translate, args.translation_model, args.thinking_budget)
