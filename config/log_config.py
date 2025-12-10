import logging
from pathlib import Path

LOG_DIR = Path(r"D:\Jiji_AI\logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "jiji.log", encoding="utf-8"),
        ],
        force=True,
    )

    logging.getLogger("faster_whisper").setLevel(logging.WARNING)