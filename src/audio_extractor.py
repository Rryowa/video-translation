import ffmpeg
import os

def extract_audio(video_path: str, output_audio_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_audio_path
