from __future__ import annotations
from pathlib import Path


CHUNK_SIZE = 500
OVERLAP = 0
ENCODING_NAME = "cl100k_base"

ROOT_PATH = Path(__file__).resolve().parent
DATA_DIR = ROOT_PATH / "data"
NOVEL_PATH = DATA_DIR / "alice.txt"


SEGMENT_SIZE = 50
JOINER = ""
PRECOLLAPSE_MIN_TOKRNS: int = 80
FILL_FLOOR: float = 0.9





# [text](url) -> keep "text"
MARKDOWN_LINK_REGEX = r"\[([^\]]+)\]\(([^)]+)\)"
# **bold** or __bold__ -> keep inner
MARKDOWN_BOLD_REGEX = r"(\*\*|__)(.+?)\1"
# *italic* or _italic_ -> keep inner
MARKDOWN_ITALIC_REGEX = r"(\*|_)(.+?)\1"
# `code` -> keep inner
MARKDOWN_CODE_REGEX = r"`([^`]+)`"
# ### Header -> keep inner
MARKDOWN_HEADER_REGEX = r"^\s{0,3}#{1,6}\s+(.+)$"