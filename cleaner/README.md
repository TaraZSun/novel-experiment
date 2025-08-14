# Text Cleaner for NLP Preprocessing

This script cleans raw text files (e.g., novels, scraped documents) to prepare them for NLP tasks such as training language models.

## Features
- Normalize line endings
- Remove decorative lines like `* * * *`
- Remove headers and page numbers
- Collapse multiple blank lines
- Merge wrapped lines into full sentences
- Remove extra spaces

## Installation
```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt  # No external dependencies needed
```

## Usage
```bash
# Run the script with your text file:
python text_cleaner.py --input alice.txt --output alice_cleaned.txt --remove-stars --remove-headers --unwrap-lines
```
## Arguments

| Flag              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `--input`, `-i`   | Path to the input text file                                   |
| `--output`, `-o`  | Path to save the cleaned text                                 |
| `--remove-stars`  | Remove lines containing only asterisks                        |
| `--remove-headers`| Remove all-caps headers and numeric page numbers              |
| `--unwrap-lines`  | Merge lines without punctuation into full sentences           |


## Example
```bash
python text_cleaner.py -i alice.txt -o alice_cleaned.txt --remove-stars --unwrap-lines
```

## Output
A cleaned .txt file ready for tokenization or chunking.

## License
MIT License