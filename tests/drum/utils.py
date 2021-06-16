from pathlib import Path

def test_data() -> Path:
    topdir = Path(__file__).parent.parent
    return topdir / 'testdata'