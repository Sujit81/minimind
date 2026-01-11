"""
Hindi Corpus Preparation Script

This script prepares Hindi text corpus for tokenizer training.
It handles:
- Unicode normalization (NFC)
- Text cleaning and deduplication
- Creating bilingual corpora (Hindi + English)
- Converting to pretraining format (JSONL)

Usage:
    python prepare_hindi_corpus.py --input_dir raw_hindi/ --output corpus_bilingual.txt
    python prepare_hindi_corpus.py --to_jsonl --input corpus_bilingual.txt
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Set
import hashlib


class HindiCorpusPreparer:
    """Prepares Hindi text corpus for tokenizer training."""

    # Patterns to clean
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    MULTIPLE_SPACES = re.compile(r'\s+')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')

    # Minimum length for a valid line (characters)
    MIN_LINE_LENGTH = 10

    # Maximum length for a valid line
    MAX_LINE_LENGTH = 10000

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.lines_processed = 0
        self.lines_kept = 0
        self.lines_removed = 0

    def normalize_text(self, text: str) -> str:
        """Normalize Unicode to NFC form."""
        return unicodedata.normalize('NFC', text)

    def clean_line(self, line: str) -> str:
        """Clean a single line of text."""
        # Strip whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            return None

        # Skip lines that are too short
        if len(line) < self.MIN_LINE_LENGTH:
            return None

        # Skip lines that are too long (likely garbage)
        if len(line) > self.MAX_LINE_LENGTH:
            return None

        # Remove URLs
        line = self.URL_PATTERN.sub(' ', line)

        # Remove emails
        line = self.EMAIL_PATTERN.sub(' ', line)

        # Normalize spaces
        line = self.MULTIPLE_SPACES.sub(' ', line)

        # Strip again
        line = line.strip()

        if len(line) < self.MIN_LINE_LENGTH:
            return None

        return line

    def is_valid_hindi_text(self, text: str) -> bool:
        """Check if text contains Devanagari characters."""
        # Check for Devanagari Unicode range
        devanagari_count = 0
        total_chars = 0

        for char in text:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:  # Devanagari block
                devanagari_count += 1
            total_chars += 1

        # At least 10% Devanagari characters
        return devanagari_count > 0 and (devanagari_count / max(total_chars, 1)) >= 0.1

    def get_hash(self, text: str) -> str:
        """Get hash of text for deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def process_file(self, input_path: Path, output_lines: List[str], mode='hindi') -> None:
        """Process a single input file."""
        print(f"Processing: {input_path}")

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    self.lines_processed += 1

                    # Clean line
                    cleaned = self.clean_line(line)
                    if not cleaned:
                        self.lines_removed += 1
                        continue

                    # Normalize Unicode
                    normalized = self.normalize_text(cleaned)

                    # Deduplicate
                    text_hash = self.get_hash(normalized)
                    if text_hash in self.seen_hashes:
                        self.lines_removed += 1
                        continue

                    # Check for Hindi content if in hindi mode
                    if mode == 'hindi' and not self.is_valid_hindi_text(normalized):
                        self.lines_removed += 1
                        continue

                    self.seen_hashes.add(text_hash)
                    output_lines.append(normalized)
                    self.lines_kept += 1

                    if self.lines_kept % 1000 == 0:
                        print(f"  Processed: {self.lines_processed:,} | Kept: {self.lines_kept:,} | Removed: {self.lines_removed:,}")

        except Exception as e:
            print(f"  Error processing {input_path}: {e}")

    def save_corpus(self, output_lines: List[str], output_path: Path) -> None:
        """Save processed corpus to file."""
        print(f"\nSaving to: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')

        print(f"Saved {len(output_lines):,} lines to {output_path}")

    def save_jsonl(self, text_lines: List[str], output_path: Path) -> None:
        """Save corpus in JSONL format for pretraining."""
        print(f"\nSaving to JSONL: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in text_lines:
                json.dump({"text": line}, f, ensure_ascii=False)
                f.write('\n')

        print(f"Saved {len(text_lines):,} lines to {output_path}")

    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("PROCESSING STATISTICS")
        print("=" * 50)
        print(f"Lines processed: {self.lines_processed:,}")
        print(f"Lines kept:      {self.lines_kept:,}")
        print(f"Lines removed:   {self.lines_removed:,}")
        print(f"Unique lines:    {len(self.seen_hashes):,}")
        print("=" * 50)


def create_sample_hindi_corpus(output_path: Path) -> None:
    """Create a sample Hindi corpus for testing."""
    sample_texts = [
        "भारत दक्षिण एशिया का एक देश है। इसकी राजधानी नई दिल्ली है और यह दुनिया की सातवीं सबसे बड़ी अर्थव्यवस्था है।",
        "महात्मा गांधी भारत के स्वतंत्रता संग्राम के प्रमुख नेता थे। उन्हें 'राष्ट्रपिता' के रूप में जाना जाता है।",
        "हिंदी भारत की राजभाषा है और यह देश की सबसे अधिक बोली जाने वाली भाषाओं में से एक है।",
        "फोटोसिंथेसिस पौधों द्वारा भोजन बनाने की प्रक्रिया है। इसमें प्रकाश, पानी और कार्बन डाइऑक्साइड का उपयोग होता है।",
        "पाइथागोरस प्रमेय के अनुसार, एक समकोण त्रिभुज में कर्ण का वर्ग अन्य दो भुजाओं के वर्गों के योग के बराबर होता है।",
        "भारत में कई नदियां बहती हैं जैसे गंगा, यमुना, ब्रह्मपुत्र, गोदावरी और कावेरी।",
        "चंद्रशेखर वेंकट रमन भारत के पहले नोबेल पुरस्कार विजेता थे। उन्हें 1930 में भौतिकी में नोबेल पुरस्कार मिला।",
        "भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) ने कई उपग्रह और चंद्रयान मिशनों को सफल बनाया है।",
        "योग भारत की एक प्राचीन परंपरा है जो शारीरिक और मानसिक स्वास्थ्य के लिए जानी जाती है।",
        "ताज महल आगरा में स्थित एक संगमरमर का मकबरा है जिसे शाहजहां ने अपनी पत्नी मुमताज महल की याद में बनवाया था।",
        "दिल्ली भारत का एक प्रमुख शहर है जिसमें नई दिल्ली (राजधानी) और पुरानी दिल्ली दोनों शामिल हैं।",
        "बॉलीवुड भारत की फिल्म उद्योग है जो मुंबई में स्थित है और दुनिया में सबसे अधिक फिल्में बनाता है।",
        "क्रिकेट भारत में सबसे लोकप्रिय खेल है और भारतीय क्रिकेट टीम ने कई अंतरराष्ट्रीय मैच जीते हैं।",
        "होली, दीवाली, ईद, और क्रिसमस भारत में धूमधाम से मनाए जाने वाले त्योहार हैं।",
        "भारत की आबादी लगभग 140 करोड़ है और यह दुनिया में सबसे अधिक आबादी वाला देश है।",
        # Hinglish examples
        "मुझे Python सीखना है क्योंकि यह एक बहुत अच्छी programming language है।",
        "क्या आप मुझे English में help कर सकते हैं? मुझे थोड़ी difficulty हो रही है।",
        "आज मैं office में बहुत busy था इसलिए late हो गया।",
        "My best friend lives in Mumbai और वहां बहुत ही अच्छी weather होती है।",
        "मैंने अभी-अभी एक new laptop खरीदा है जिसमें 16GB RAM है।",
    ]

    # Duplicate some texts to simulate larger corpus
    extended_texts = sample_texts * 50

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in extended_texts:
            f.write(text + '\n')

    print(f"Created sample Hindi corpus with {len(extended_texts)} lines: {output_path}")


def create_bilingual_corpus(hindi_file: Path, english_file: Path, output_path: Path, ratio: float = 0.7) -> None:
    """Create a bilingual corpus with Hindi and English mixed."""
    hindi_lines = []
    english_lines = []

    if hindi_file.exists():
        with open(hindi_file, 'r', encoding='utf-8') as f:
            hindi_lines = [line.strip() for line in f if line.strip()]

    if english_file.exists():
        with open(english_file, 'r', encoding='utf-8') as f:
            english_lines = [line.strip() for line in f if line.strip()]

    total_hindi = int(len(hindi_lines) * ratio)
    total_english = int(len(english_lines) * (1 - ratio))

    output_lines = hindi_lines[:total_hindi] + english_lines[:total_english]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f"Created bilingual corpus: {len(output_lines)} lines ({total_hindi} Hindi, {total_english} English)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Hindi corpus for tokenizer training"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='raw_hindi/',
        help='Input directory containing Hindi text files'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Single input file to process'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='dataset/hindi/corpus_raw.txt',
        help='Output corpus file'
    )
    parser.add_argument(
        '--to_jsonl',
        action='store_true',
        help='Convert output to JSONL format for pretraining'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['hindi', 'english', 'mixed'],
        default='hindi',
        help='Processing mode: hindi (require Devanagari), english, or mixed'
    )
    parser.add_argument(
        '--bilingual',
        action='store_true',
        help='Create bilingual corpus (Hindi + English)'
    )
    parser.add_argument(
        '--hindi_ratio',
        type=float,
        default=0.7,
        help='Ratio of Hindi in bilingual corpus (default: 0.7)'
    )
    parser.add_argument(
        '--english_file',
        type=str,
        help='English corpus file for bilingual mixing'
    )
    parser.add_argument(
        '--create_sample',
        action='store_true',
        help='Create sample Hindi corpus for testing'
    )

    args = parser.parse_args()

    preparer = HindiCorpusPreparer()
    output_lines = []

    # Create sample corpus
    if args.create_sample:
        sample_path = Path('dataset/hindi/corpus_raw.txt')
        create_sample_hindi_corpus(sample_path)

        # Also create bilingual sample
        bilingual_path = Path('dataset/hindi/corpus_bilingual.txt')
        create_sample_hindi_corpus(bilingual_path)

        print("\nSample corpora created successfully!")
        preparer.print_stats()
        return

    # Process single file
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.exists():
            preparer.process_file(input_path, output_lines, mode=args.mode)
        else:
            print(f"Error: Input file '{args.input_file}' not found!")
            return

    # Process directory
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if input_dir.exists() and input_dir.is_dir():
            for file_path in input_dir.glob('**/*.txt'):
                preparer.process_file(file_path, output_lines, mode=args.mode)
        else:
            print(f"Error: Input directory '{args.input_dir}' not found!")
            return

    # Create bilingual corpus
    if args.bilingual:
        if args.english_file:
            create_bilingual_corpus(
                Path(args.output),
                Path(args.english_file),
                Path(args.output).parent / 'corpus_bilingual.txt',
                args.hindi_ratio
            )

    # Save output
    if output_lines:
        output_path = Path(args.output)

        if args.to_jsonl:
            jsonl_path = output_path.parent / (output_path.stem + '.jsonl')
            preparer.save_jsonl(output_lines, jsonl_path)
        else:
            preparer.save_corpus(output_lines, output_path)

        preparer.print_stats()
    else:
        print("No lines processed. Please check your input files.")


if __name__ == "__main__":
    main()
