from pathlib import Path


def test_data() -> Path:
    top_dir = Path(__file__).parent.parent
    return top_dir / "testdata"
