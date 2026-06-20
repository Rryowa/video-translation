import math

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def is_cjk_char(char: str) -> bool:
    return any(u'\u4e00' <= c <= u'\u9fff' or u'\u3040' <= c <= u'\u30ff' or u'\uac00' <= c <= u'\ud7af' for c in char)

def join_words(items) -> str:
    text = ""
    for i, item in enumerate(items):
        txt = item.text
        if i > 0:
            prev_txt = items[i-1].text
            if len(prev_txt) > 0 and len(txt) > 0:
                if not (is_cjk_char(prev_txt[-1]) and is_cjk_char(txt[0])):
                    text += " "
        text += txt
    return text

def group_alignments(items, max_words=8, max_chars=30, max_gap=0.8, end_padding=0.0):
    segments = []
    current_segment = []
    
    for item in items:
        should_split = False
        if current_segment:
            prev_item = current_segment[-1]
            gap = item.start_time - prev_item.end_time
            if gap > max_gap:
                should_split = True
            elif len(current_segment) >= max_words:
                should_split = True
            elif sum(len(x.text) for x in current_segment) + len(item.text) > max_chars:
                should_split = True
                
        if should_split and current_segment:
            segments.append({
                "start": current_segment[0].start_time,
                "end": current_segment[-1].end_time,
                "text": join_words(current_segment)
            })
            current_segment = []
            
        current_segment.append(item)
        
    if current_segment:
        segments.append({
            "start": current_segment[0].start_time,
            "end": current_segment[-1].end_time,
            "text": join_words(current_segment)
        })
        
    # Apply end padding to give the viewer extra time to read the subtitles
    if end_padding > 0:
        for i in range(len(segments)):
            current_end = segments[i]["end"]
            padded_end = current_end + end_padding
            if i < len(segments) - 1:
                next_start = segments[i+1]["start"]
                # Keep a tiny 0.1s gap before the next segment starts
                segments[i]["end"] = min(padded_end, next_start - 0.1)
            else:
                segments[i]["end"] = padded_end
        
    return segments

def generate_srt(segments: list, output_srt_path: str):
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text'].strip()}\n\n")
