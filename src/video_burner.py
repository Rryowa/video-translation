import ffmpeg
import os

def escape_windows_path(path: str) -> str:
    path = os.path.abspath(path)
    path = path.replace('\\', '/')
    if ':' in path:
        path = path.replace(':', '\\:')
    return path

def burn_subtitles(video_path: str, srt_path: str, output_video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
    escaped_srt = escape_windows_path(srt_path)
    
    (
        ffmpeg
        .input(video_path)
        .output(output_video_path, vf=f"subtitles='{escaped_srt}'", vcodec='h264_nvenc', acodec='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_video_path
