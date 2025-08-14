# chunkers/llm_chunker.py
from __future__ import annotations
import json
import re
import os
from typing import List, Dict, Optional
from pathlib import Path
import base64
from .base_chunker import BaseChunker
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("`openai` not installed. Run: pip install openai")

try:
    import yaml  # PyYAML
except ImportError:
    raise ImportError("`PyYAML` not installed. Run: pip install pyyaml")


def _load_yaml(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "system" not in data or "user" not in data:
        raise ValueError(f"Prompt YAML must contain 'system' and 'user' keys: {p}")
    return data  # type: ignore[return-value]


def _render_template(tmpl: str, **vars) -> str:
    def repl(m):
        key = m.group(1).strip()
        return str(vars.get(key, ""))
    return re.sub(r"\{\{\s*(.*?)\s*\}\}", repl, tmpl)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


class LLMChunker(BaseChunker):
    """
    LLM-based chunker that uses an external YAML prompt with keys: 'system' and 'user'.
    The model must return JSON:
      {"chunks": [{"text": "...", "note": "..."}]}
    Note: `overlap_sentences` is advisory to the LLM via prompt only.

    This version batches long inputs internally to avoid provider token limits.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",  # Groq default; override if using OpenAI
        size: int = 500,
        prompt_path: str | Path = "prompts/llm_chunker_prompt.yaml",
        api_key: Optional[str] = None,
        overlap_sentences: int = 1,
        max_tokens_headroom: int = 50,
        # NEW: internal batching
        llm_batch_chars: Optional[int] = 3000,   # ~safe for Groq free tier
        llm_max_batches: Optional[int] = None,   # cap for testing/debug
    ):
        super().__init__(size)
        self.model = model
        self.prompt_path = Path(prompt_path)
        self.overlap_sentences = overlap_sentences
        self.max_tokens = size + max_tokens_headroom

        # Prefer Groq if GROQ_API_KEY present; fallback to OpenAI
        groq_key = api_key or os.getenv("GROQ_API_KEY")
        if groq_key:
            self.client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            openai_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_key)

        self.prompt = _load_yaml(self.prompt_path)

        # batching settings
        # self.llm_batch_chars = int(llm_batch_chars) if llm_batch_chars else 3000
        self.llm_batch_chars = int(llm_batch_chars or 2000)
        self.llm_max_batches = llm_max_batches

    # ------- internal helpers -------

    def _split_into_batches(self, text: str) -> List[str]:
        """
        Robust paragraph-aware splitter with hard-wrap fallback.
        - If there are no double newlines, or a paragraph is longer than max_chars,
        we hard-split by characters to keep each batch under the limit.
        """
        max_chars = self.llm_batch_chars  # already an int from __init__
        if len(text) <= max_chars:
            return [text]

        paras = text.split("\n\n")
        # Fallback: no paragraph breaks -> hard-wrap the whole text
        if len(paras) == 1:
            s = paras[0]
            return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]

        batches: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for p in paras:
            # If a single paragraph is too large, first flush current, then hard-split this paragraph.
            if len(p) + 2 > max_chars:  # +2 for the two newlines we re-insert normally
                if cur:
                    batches.append("\n\n".join(cur))
                    cur, cur_len = [], 0
                # hard-wrap the long paragraph
                for i in range(0, len(p), max_chars):
                    chunk = p[i : i + max_chars]
                    batches.append(chunk)
                    if self.llm_max_batches and len(batches) >= self.llm_max_batches:
                        return batches[: self.llm_max_batches]
                continue

            # Normal paragraph packing
            p_len = len(p) + 2
            if cur and (cur_len + p_len) > max_chars:
                batches.append("\n\n".join(cur))
                cur, cur_len = [p], p_len
                if self.llm_max_batches and len(batches) >= self.llm_max_batches:
                    return batches[: self.llm_max_batches]
            else:
                cur.append(p)
                cur_len += p_len

        if cur and (not self.llm_max_batches or len(batches) < self.llm_max_batches):
            batches.append("\n\n".join(cur))

        if self.llm_max_batches:
            batches = batches[: self.llm_max_batches]

        return batches

   
    def _call_llm(self, text: str) -> Dict:
        system_msg = _render_template(
            self.prompt["system"],
            target_tokens=self.chunk_size,
            max_tokens=self.max_tokens,
            overlap_sentences=self.overlap_sentences,
        )
        user_msg = _render_template(
            self.prompt["user"],
            text=text,
            target_tokens=self.chunk_size,
            max_tokens=self.max_tokens,
            overlap_sentences=self.overlap_sentences,
        )

        # Try JSON mode; fallback if unsupported
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
            )

        content = (resp.choices[0].message.content or "").strip()
        content = _strip_code_fences(content)

        def _try_parse_json(s: str) -> Dict:
            return json.loads(s)

        def _sanitize_json(s: str) -> str:
            # keep only the first {...}
            m = re.search(r"\{.*\}", s, flags=re.S)
            if m:
                s = m.group(0)
            s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("\ufeff", "")
            s = re.sub(r"\bNone\b", "null", s)
            s = re.sub(r"\bTrue\b", "true", s)
            s = re.sub(r"\bFalse\b", "false", s)
            s = re.sub(r'(?P<pre>[\{\s,])\'(?P<key>[^\'"\n\r\t]+)\'\s*:', r'\g<pre>"\g<key>":', s)
            s = re.sub(r':\s*\'([^\'"]*?)\'', r': "\1"', s)
            s = re.sub(r",\s*([}\]])", r"\1", s)
            return s

        try:
            data = _try_parse_json(content)
        except json.JSONDecodeError:
            data = _try_parse_json(_sanitize_json(content))

        # Expect {"chunks":[{"text_b64": "...", "note": ...}, ...]}
        if not isinstance(data, dict) or "chunks" not in data or not isinstance(data["chunks"], list):
            snippet = content[:500].replace("\n", "\\n")
            raise ValueError(f"LLM returned JSON without a 'chunks' list. First 500 chars: {snippet}")

        # Decode base64 -> "text"; keep backward-compat if model returned "text"
        for i, item in enumerate(data["chunks"]):
            if "text_b64" in item:
                try:
                    raw = base64.b64decode(item["text_b64"]).decode("utf-8", errors="replace")
                except Exception:
                    raise ValueError(f"Invalid base64 in chunk #{i+1}")
                item["text"] = raw
                item.pop("text_b64", None)
            # normalize note to None if missing
            if "note" not in item or item["note"] == "":
                item["note"] = None

        return data



    # ------- public API -------

    def chunk(self, text: str) -> List[Dict]:
        """
        If the text is long, automatically split into batches and call the LLM per batch.
        Merge all returned chunks into one list with monotonically increasing ids.
        """
        batches = self._split_into_batches(text)
        all_out: List[Dict] = []
        cid = 1

        for batch_text in batches:
            data = self._call_llm(batch_text)
            chunks_json = data.get("chunks", []) or []
            for item in chunks_json:
                t = (item.get("text") or "").strip()
                if not t:
                    continue
                all_out.append({
                    "id": cid,
                    "text": t,
                    "token_count": self.count_tokens(t),
                    "strategy": "llm",
                    "note": item.get("note"),
                })
                cid += 1

        return all_out
