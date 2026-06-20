from src.srt_generator import format_time, generate_srt, group_alignments

class MockItem:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

def test_format_time():
    assert format_time(1.234) == "00:00:01,234"
    assert format_time(3661.5) == "01:01:01,500"

def test_generate_srt(tmp_path):
    segments = [{"start": 0.5, "end": 1.5, "text": "Hello world"}]
    out_file = tmp_path / "out.srt"
    generate_srt(segments, str(out_file))
    content = out_file.read_text(encoding="utf-8")
    assert "1\n" in content
    assert "00:00:00,500 --> 00:00:01,500" in content
    assert "Hello world" in content

def test_group_alignments():
    items = [
        MockItem("Hello", 0.1, 0.5),
        MockItem("world", 0.6, 1.0),
        MockItem("this", 1.8, 2.2),
        MockItem("is", 2.3, 2.5),
        MockItem("test", 2.6, 3.0),
    ]
    # Expect split at "this" due to gap (1.8 - 1.0 = 0.8 > 0.5)
    segs = group_alignments(items, max_words=3, max_gap=0.5)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world"
    assert segs[1]["text"] == "this is test"
