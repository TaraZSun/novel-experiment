#!/usr/bin/env python3
import re
import argparse

class TextCleaner:
    def __init__(self, remove_stars=True, remove_headers=True, unwrap_lines=True):
        self.remove_stars = remove_stars
        self.remove_headers = remove_headers
        self.unwrap_lines = unwrap_lines

    def clean(self, text):
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove lines with only * * * etc.
        if self.remove_stars:
            text = re.sub(r'^\s*(?:\*\s*){3,}\s*$', '', text, flags=re.MULTILINE)

        # Remove all-caps headers and numeric page numbers
        if self.remove_headers:
            lines = []
            for line in text.split("\n"):
                s = line.strip()
                if re.fullmatch(r"[A-Z\s]{5,}", s):
                    continue
                if re.fullmatch(r"\d+", s):
                    continue
                lines.append(line)
            text = "\n".join(lines)

        # Collapse multiple blank lines
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Merge broken lines (unwrap)
        if self.unwrap_lines:
            text = re.sub(r"(?<![.?!:;\"\'])\n(?!\n)", " ", text)

        # Remove extra spaces
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def clean_file(self, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned_text = self.clean(text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"✅ Cleaned text saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw text files for NLP preprocessing.")
    parser.add_argument("--input", "-i", required=True, help="Path to input text file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save cleaned text file.")
    parser.add_argument("--remove-stars", action="store_true", help="Remove lines containing only asterisks.")
    parser.add_argument("--remove-headers", action="store_true", help="Remove all-caps headers and page numbers.")
    parser.add_argument("--unwrap-lines", action="store_true", help="Merge lines without punctuation at the end.")

    args = parser.parse_args()

    cleaner = TextCleaner(
        remove_stars=args.remove_stars,
        remove_headers=args.remove_headers,
        unwrap_lines=args.unwrap_lines
    )
    cleaner.clean_file(args.input, args.output)
