from bs4 import BeautifulSoup
import re
import logging
from constants import (
    MARKDOWN_LINK_REGEX,
    MARKDOWN_BOLD_REGEX,
    MARKDOWN_ITALIC_REGEX,
    MARKDOWN_CODE_REGEX,
    MARKDOWN_HEADER_REGEX
)
logger = logging.getLogger(__name__)
_MARKDOWN_RULES: list[tuple[re.Pattern,str]] = [
    (re.compile(MARKDOWN_LINK_REGEX), r"\1"),
    (re.compile(MARKDOWN_BOLD_REGEX), r"\2"),
    (re.compile(MARKDOWN_ITALIC_REGEX), r"\2"),
    (re.compile(MARKDOWN_CODE_REGEX), r"\1"),
    (re.compile(MARKDOWN_HEADER_REGEX), r"\1"),
]



def default_preprocess(text:str, *, clean_html: bool = True, clean_markdown: bool = True)-> str:
    if clean_html:
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            logger.exception("Failed to clean HTML with BeautifulSoup.")
    
    if clean_markdown:
        for pattern_obj, replacement in _MARKDOWN_RULES:
            text = pattern_obj.sub(replacement, text)
    return " ".join(text.split())



        

