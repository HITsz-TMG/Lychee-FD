from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AVATAR_IMAGE_PATH = str(
    (PROJECT_ROOT / "assets" / "avatar" / "default.png").resolve()
)
DEFAULT_AVATAR_PROMPT = "A person is having a natural conversation with the user."
