"""
Hindi Tokenizer Evaluation Script

This script evaluates a tokenizer's performance on Hindi text, measuring:
- Average tokens per word (efficiency)
- Devanagari character coverage
- Byte fallback rate
- Special token preservation

Usage:
    python evaluate_hindi_tokenizer.py --tokenizer_path ../model_hindi
    python evaluate_hindi_tokenizer.py --tokenizer_path ../model --test
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class HindiTokenizerEvaluator:
    """Evaluates tokenizer performance on Hindi text."""

    # Sample Hindi texts for testing
    SAMPLE_TEXTS = [
        "भारत दक्षिण एशिया का एक देश है।",
        "महात्मा गांधी भारत के स्वतंत्रता संग्राम के प्रमुख नेता थे।",
        "फोटोसिंथेसिस पौधों द्वारा भोजन बनाने की प्रक्रिया है।",
        "पाइथागोरस प्रमेय: a² + b² = c²",
        "Python में एक फाइबोनैकी फंक्शन:",
        "def fibonacci(n):",
        "    if n <= 1: return n",
        "    return fibonacci(n-1) + fibonacci(n-2)",
        "१, २, ३, ४, ५, ६, ७, ८, ९, ० - हिंदी अंक",
        "वर्षा ऋषि के ऋण से ऋजु अवस्था में आया।",  # Contains conjuncts
        "क्षत्रिय वीर श्रेष्ठ ज्ञानी द्वारा",  # Contains complex conjuncts
    ]

    # Common Hindi words for efficiency testing
    COMMON_WORDS = [
        "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "पर", "को", "से",
        "एक", "और", "यह", "वह", "भी", "ने", "कि", "जो", "हो", "गया", "कर",
        "सकता", "लिए", "बहुत", "जाता", "होता", "बना", "रहा", "रही", "कहा",
    ]

    # Devanagari characters to check
    VOWELS = list("अआइईउऊऋएऐओऔ")
    CONSONANTS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह")
    MATRAS = list("ािीुूृेैोौ")
    NUMERALS = list("०१२३४५६७८९")
    SPECIAL_CHARS = ["ं", "ः", "ँ", "।", "॥"]
    CONJUNCTS = ["क्ष", "त्र", "ज्ञ", "श्र", "द्व"]

    # Expected special tokens
    EXPECTED_SPECIAL_TOKENS = {
        0: "<|endoftext|>",  # PAD/UNK
        1: "<|im_start|>",   # BOS
        2: "<|im_end|>",     # EOS
    }

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = None
        self.vocab_size = 0
        self.results = {}

    def load_tokenizer(self):
        """Load the tokenizer from the given path."""
        try:
            from transformers import AutoTokenizer

            print(f"Loading tokenizer from: {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
                trust_remote_code=True
            )
            self.vocab_size = len(self.tokenizer)
            print(f"✓ Tokenizer loaded successfully! Vocab size: {self.vocab_size}")
            return True

        except Exception as e:
            print(f"✗ Error loading tokenizer: {e}")
            return False

    def check_special_tokens(self) -> Dict:
        """Check if special tokens are correctly configured."""
        print("\n" + "=" * 60)
        print("SPECIAL TOKENS CHECK")
        print("=" * 60)

        result = {
            'correct': True,
            'details': {}
        }

        for token_id, expected_token in self.EXPECTED_SPECIAL_TOKENS.items():
            # Get actual token from tokenizer
            actual_token = self.tokenizer.decode([token_id])

            # Check by convert_tokens_to_ids as well
            decoded_id = self.tokenizer.convert_tokens_to_ids(expected_token)

            is_correct = actual_token == expected_token
            status = "✓" if is_correct else "✗"

            print(f"{status} Token ID {token_id}: Expected '{expected_token}', Got '{actual_token}' (decoded_id: {decoded_id})")

            result['details'][token_id] = {
                'expected': expected_token,
                'actual': actual_token,
                'correct': is_correct
            }

            if not is_correct:
                result['correct'] = False

        return result

    def check_char_coverage(self) -> Dict:
        """Check Devanagari character coverage in vocabulary."""
        print("\n" + "=" * 60)
        print("DEVANAGARI CHARACTER COVERAGE")
        print("=" * 60)

        categories = {
            'Vowels': self.VOWELS,
            'Consonants': self.CONSONANTS,
            'Matras': self.MATRAS,
            'Numerals': self.NUMERALS,
            'Special': self.SPECIAL_CHARS,
            'Conjuncts': self.CONJUNCTS,
        }

        result = {}

        for category, chars in categories.items():
            covered = []
            not_covered = []

            for char in chars:
                # Try to tokenize the character
                tokens = self.tokenizer.tokenize(char)

                # Check if character is a single token or efficiently tokenized
                if len(tokens) == 1 or (len(tokens) <= 2 and not any(t.startswith('<0x') for t in tokens)):
                    covered.append(char)
                else:
                    not_covered.append(char)

            percentage = (len(covered) / len(chars) * 100) if chars else 0
            status = "✓" if percentage >= 90 else ("~" if percentage >= 70 else "✗")

            print(f"{status} {category:15} {len(covered):3}/{len(chars):<3} ({percentage:5.1f}%) ", end='')

            if not_covered:
                print(f"Missing/Inefficient: {' '.join(not_covered[:5])}", end='')
                if len(not_covered) > 5:
                    print(f" +{len(not_covered)-5} more", end='')
            print()

            result[category] = {
                'total': len(chars),
                'covered': len(covered),
                'not_covered': not_covered,
                'percentage': percentage
            }

        return result

    def measure_token_efficiency(self) -> Dict:
        """Measure average tokens per word for Hindi text."""
        print("\n" + "=" * 60)
        print("TOKEN EFFICIENCY (tokens per word)")
        print("=" * 60)

        # Test on common words
        word_tokens = []
        for word in self.COMMON_WORDS:
            tokens = self.tokenizer.tokenize(word)
            word_tokens.append(len(tokens))

        avg_tokens_per_word = sum(word_tokens) / len(word_tokens) if word_tokens else 0

        print(f"Common Hindi words: {avg_tokens_per_word:.2f} tokens/word (target: < 2.0)")

        # Show some examples
        print("\nExamples:")
        for word in self.COMMON_WORDS[:10]:
            tokens = self.tokenizer.tokenize(word)
            print(f"  '{word:12}' -> {tokens} ({len(tokens)} tokens)")

        # Test on sample sentences
        sentence_tokens = []
        for text in self.SAMPLE_TEXTS[:5]:
            tokens = self.tokenizer.tokenize(text)
            words = len(text.split())
            sentence_tokens.append(len(tokens) / max(words, 1))

        avg_tokens_per_sentence = sum(sentence_tokens) / len(sentence_tokens) if sentence_tokens else 0

        print(f"\nSample sentences: {avg_tokens_per_sentence:.2f} tokens/word")

        return {
            'avg_tokens_per_word': avg_tokens_per_word,
            'avg_tokens_per_sentence': avg_tokens_per_sentence,
            'target': 2.0,
            'pass': avg_tokens_per_word < 2.0
        }

    def check_byte_fallback(self) -> Dict:
        """Check for byte-level fallback usage (indicates poor coverage)."""
        print("\n" + "=" * 60)
        print("BYTE FALLBACK ANALYSIS")
        print("=" * 60)

        byte_fallback_count = 0
        total_tokens = 0

        # Test on sample texts
        for text in self.SAMPLE_TEXTS:
            tokens = self.tokenizer.tokenize(text)
            for token in tokens:
                total_tokens += 1
                # Byte fallback tokens look like <0xXX>
                if token.startswith('<0x') or token.startswith('▁<0x'):
                    byte_fallback_count += 1

        fallback_rate = (byte_fallback_count / total_tokens * 100) if total_tokens > 0 else 0
        status = "✓" if fallback_rate < 1 else ("~" if fallback_rate < 5 else "✗")

        print(f"{status} Byte fallback rate: {fallback_rate:.2f}% (target: < 1%)")

        return {
            'byte_fallback_count': byte_fallback_count,
            'total_tokens': total_tokens,
            'fallback_rate': fallback_rate,
            'target': 1.0,
            'pass': fallback_rate < 1.0
        }

    def check_vocabulary_size(self) -> Dict:
        """Check vocabulary size and recommendation."""
        print("\n" + "=" * 60)
        print("VOCABULARY SIZE")
        print("=" * 60)

        print(f"Current vocab size: {self.vocab_size}")

        # Recommendations
        if self.vocab_size < 8000:
            recommendation = "Consider increasing to 8000-10000 for better Hindi coverage"
            status = "~"
        elif 8000 <= self.vocab_size <= 12000:
            recommendation = "Good size for Hindi+English bilingual support"
            status = "✓"
        else:
            recommendation = "Large vocabulary - ensure sufficient training data"
            status = "~"

        print(f"{status} {recommendation}")

        return {
            'vocab_size': self.vocab_size,
            'recommendation': recommendation
        }

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        print("\n" + "=" * 60)
        print("OVERALL ASSESSMENT")
        print("=" * 60)

        # Collect all results
        self.results['special_tokens'] = self.check_special_tokens()
        self.results['char_coverage'] = self.check_char_coverage()
        self.results['efficiency'] = self.measure_token_efficiency()
        self.results['byte_fallback'] = self.check_byte_fallback()
        self.results['vocab_size'] = self.check_vocabulary_size()

        # Calculate overall score
        passed_checks = 0
        total_checks = 0

        # Special tokens check
        if self.results['special_tokens']['correct']:
            passed_checks += 1
        total_checks += 1

        # Efficiency check
        if self.results['efficiency']['pass']:
            passed_checks += 1
        total_checks += 1

        # Byte fallback check
        if self.results['byte_fallback']['pass']:
            passed_checks += 1
        total_checks += 1

        # Overall character coverage (average percentage)
        char_coverage_results = self.results['char_coverage']
        avg_coverage = sum(
            cat['percentage'] for cat in char_coverage_results.values()
        ) / len(char_coverage_results) if char_coverage_results else 0

        if avg_coverage >= 90:
            passed_checks += 1
        total_checks += 1

        overall_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        status = "✓ EXCELLENT" if overall_percentage >= 75 else ("~ ACCEPTABLE" if overall_percentage >= 50 else "✗ NEEDS IMPROVEMENT")
        print(f"\n{status} - {passed_checks}/{total_checks} checks passed ({overall_percentage:.0f}%)")

        print("\n" + "=" * 60)

        return status

    def save_report(self, output_path: str = None):
        """Save evaluation report to JSON file."""
        if output_path is None:
            output_path = self.tokenizer_path.parent / "tokenizer_evaluation_report.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hindi tokenizer performance"
    )
    parser.add_argument(
        '--tokenizer_path', '-t',
        type=str,
        default='../model_hindi',
        help='Path to tokenizer directory'
    )
    parser.add_argument(
        '--save_report',
        action='store_true',
        help='Save evaluation report to JSON'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for evaluation report'
    )

    args = parser.parse_args()

    evaluator = HindiTokenizerEvaluator(args.tokenizer_path)

    if not evaluator.load_tokenizer():
        return 1

    status = evaluator.generate_test_report()

    if args.save_report:
        evaluator.save_report(args.output)

    # Return exit code
    if "EXCELLENT" in status:
        return 0
    elif "ACCEPTABLE" in status:
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())
