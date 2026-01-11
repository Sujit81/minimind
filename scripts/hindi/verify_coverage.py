"""
Devanagari Character Coverage Verification Script

This script verifies that Hindi corpus text contains all necessary Devanagari characters
including vowels, consonants, matras, conjuncts, punctuation, and numerals.

Usage:
    python verify_coverage.py --input_file corpus_raw.txt
    python verify_coverage.py --input_file corpus_raw.txt --detailed
"""

import argparse
import unicodedata
from collections import Counter
from pathlib import Path


class DevanagariCoverageChecker:
    """Checks Devanagari character coverage in Hindi text corpus."""

    # Essential Devanagari characters that should be present
    VOWELS = "अ आ इ ई उ ऊ ऋ ए ऐ ओ औ".split()
    CONSONANTS = "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल ळ व श ष स ह".split()
    MATRAS = "ा ि ी ु ू ृ े ै ो ौ".split()
    CONJUNCTS = ["क्ष", "त्र", "ज्ञ", "श्र", "द्व", "स्व", "न्त", "त्त"]
    PUNCTUATION = "। ॥ ऽ".split()
    NUMERALS = "० १ २ ३ ४ ५ ६ ७ ८ ९".split()
    DIACRITICS = "ं ः ँ".split()  # Anusvara, Visarga, Chandrabindu

    # Unicode ranges for Devanagari
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    DEVANAGARI_EXTENDED = (0xA8E0, 0xA8FF)

    def __init__(self):
        self.missing_chars = set()
        self.present_chars = set()
        self.char_frequency = Counter()
        self.total_chars = 0
        self.devanagari_chars = 0

    def is_devanagari(self, char):
        """Check if a character is in Devanagari script."""
        code = ord(char)
        return (self.DEVANAGARI_RANGE[0] <= code <= self.DEVANAGARI_RANGE[1] or
                self.DEVANAGARI_EXTENDED[0] <= code <= self.DEVANAGARI_EXTENDED[1])

    def check_coverage(self, text):
        """Check coverage of Devanagari characters in the given text."""
        # Normalize text to NFC form
        text = unicodedata.normalize('NFC', text)

        self.total_chars = len(text)

        # Count character frequencies
        for char in text:
            if self.is_devanagari(char):
                self.devanagari_chars += 1
                self.char_frequency[char] += 1

        # Check each category
        results = {
            'vowels': self._check_chars(self.VOWELS, text),
            'consonants': self._check_chars(self.CONSONANTS, text),
            'matras': self._check_chars(self.MATRAS, text),
            'conjuncts': self._check_chars(self.CONJUNCTS, text),
            'punctuation': self._check_chars(self.PUNCTUATION, text),
            'numerals': self._check_chars(self.NUMERALS, text),
            'diacritics': self._check_chars(self.DIACRITICS, text),
        }

        return results

    def _check_chars(self, char_list, text):
        """Check if characters from a list are present in text."""
        result = {
            'total': len(char_list),
            'present': [],
            'missing': []
        }

        for char in char_list:
            if char in text:
                result['present'].append(char)
                self.present_chars.add(char)
            else:
                result['missing'].append(char)
                self.missing_chars.add(char)

        return result

    def print_report(self, results, detailed=False):
        """Print coverage report."""
        print("=" * 60)
        print("DEVANAGARI CHARACTER COVERAGE REPORT")
        print("=" * 60)
        print(f"\nTotal characters in corpus: {self.total_chars:,}")
        print(f"Devanagari characters: {self.devanagari_chars:,} ({self.devanagari_chars/max(1,self.total_chars)*100:.1f}%)")
        print()

        overall_total = 0
        overall_present = 0

        # Category reports
        for category, data in results.items():
            total = data['total']
            present = len(data['present'])
            missing = len(data['missing'])
            percentage = (present / total * 100) if total > 0 else 0

            overall_total += total
            overall_present += present

            status = "✓" if percentage >= 90 else ("~" if percentage >= 70 else "✗")
            print(f"{status} {category.upper():15} {present:3}/{total:<3} ({percentage:5.1f}%)", end='')

            if missing > 0 and detailed:
                print(f" Missing: {' '.join(data['missing'][:5])}", end='')
                if len(data['missing']) > 5:
                    print(f" +{len(data['missing'])-5} more", end='')
            print()

        # Overall summary
        print("-" * 60)
        overall_percentage = (overall_present / overall_total * 100) if overall_total > 0 else 0
        print(f"OVERALL: {overall_present}/{overall_total} ({overall_percentage:.1f}%)")

        if self.missing_chars and detailed:
            print("\n" + "=" * 60)
            print("MISSING CHARACTERS:")
            print("=" * 60)
            for char in sorted(self.missing_chars):
                name = unicodedata.name(char, "UNKNOWN")
                print(f"  '{char}' U+{ord(char):04X} - {name}")

        if detailed and self.present_chars:
            print("\n" + "=" * 60)
            print("CHARACTER FREQUENCY (Top 20):")
            print("=" * 60)
            for char, freq in self.char_frequency.most_common(20):
                name = unicodedata.name(char, "UNKNOWN")
                print(f"  '{char}' {freq:6,} - {name}")

        print("=" * 60)

        return overall_percentage


def check_file(file_path, detailed=False):
    """Check Devanagari coverage in a file."""
    checker = DevanagariCoverageChecker()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            print(f"Warning: File '{file_path}' is empty!")
            return 0

        results = checker.check_coverage(text)
        coverage = checker.print_report(results, detailed)

        return coverage

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        return 0
    except UnicodeDecodeError:
        print(f"Error: Could not decode '{file_path}' as UTF-8!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify Devanagari character coverage in Hindi corpus"
    )
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        default='dataset/hindi/corpus_raw.txt',
        help='Input Hindi corpus file (default: dataset/hindi/corpus_raw.txt)'
    )
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed report with missing characters and frequencies'
    )

    args = parser.parse_args()

    print(f"Checking file: {args.input_file}")
    print()

    coverage = check_file(args.input_file, args.detailed)

    # Return exit code based on coverage
    if coverage >= 90:
        print("\n✓ Excellent coverage!")
        return 0
    elif coverage >= 70:
        print("\n~ Acceptable coverage, but some characters missing.")
        return 1
    else:
        print("\n✗ Poor coverage! More diverse Hindi text needed.")
        return 2


if __name__ == "__main__":
    exit(main())
