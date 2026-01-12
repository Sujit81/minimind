# Hindi Tokenizer Evaluation Script for MiniMind
# Comprehensive evaluation of Hindi tokenizer covering all linguistic aspects
# हिंदी टोकननाइज़र का व्यापक मूल्यांकन
#
# Tests:
# - Basic encoding/decoding roundtrip
# - Special tokens
# - Conjunct preservation (संयुक्त अक्षर)
# - Verb morphology (क्रिया रूप)
# - Noun morphology (संज्ञा रूप)
# - Postpositions (परसर्ग)
# - Pronouns (सर्वनाम)
# - Numbers (संख्याएँ)
# - Prefixes/Suffixes (उपसर्ग/प्रत्यय)
# - Token efficiency metrics
# - Bilingual (Hindi-English) handling
# - Unicode normalization
# - Chat template
# - Edge cases

import os
import sys
import json
import unicodedata
import io
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Fix Windows console encoding for Hindi output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sentencepiece as spm

# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Conjuncts that should be preserved as single units
CONJUNCTS = {
    "Common": ["क्ष", "त्र", "ज्ञ", "श्र", "द्व", "द्य", "त्य", "स्त", "स्थ", "न्य"],
    "क-series": ["क्क", "क्त", "क्र", "क्ल", "क्व", "क्ष"],
    "त-series": ["त्त", "त्न", "त्म", "त्य", "त्र", "त्व", "त्स"],
    "द-series": ["द्ग", "द्द", "द्ध", "द्न", "द्म", "द्य", "द्र", "द्व"],
    "न-series": ["न्त", "न्द", "न्ध", "न्न", "न्म", "न्य", "न्र", "न्व"],
    "स-series": ["स्क", "स्त", "स्थ", "स्न", "स्प", "स्म", "स्य", "स्र", "स्व"],
    "श-series": ["श्च", "श्न", "श्म", "श्य", "श्र", "श्ल", "श्व"],
}

# Verb morphology test cases
VERB_MORPHOLOGY = {
    "Infinitive (मूल रूप)": {
        "forms": ["करना", "जाना", "खाना", "पीना", "सोना", "देखना", "सुनना", "बोलना"],
        "expected_suffixes": ["ना"],
    },
    "Present Habitual (सामान्य वर्तमान)": {
        "forms": ["करता", "करती", "करते", "जाता", "जाती", "जाते"],
        "expected_suffixes": ["ता", "ती", "ते"],
    },
    "Present Continuous (अपूर्ण वर्तमान)": {
        "forms": ["कर रहा है", "कर रही है", "कर रहे हैं", "जा रहा है"],
        "expected_suffixes": ["रहा", "रही", "रहे"],
    },
    "Past (भूतकाल)": {
        "forms": ["किया", "गया", "खाया", "देखा", "सुना", "बोला"],
        "expected_suffixes": ["या", "ा"],
    },
    "Past Perfect (पूर्ण भूत)": {
        "forms": ["किया था", "गया था", "खाया था", "की थी", "गई थी"],
        "expected_suffixes": ["था", "थी", "थे"],
    },
    "Future (भविष्य काल)": {
        "forms": ["करेगा", "करेगी", "करेंगे", "जाएगा", "जाएगी", "जाएँगे"],
        "expected_suffixes": ["गा", "गी", "गे"],
    },
    "Imperative (आज्ञार्थ)": {
        "forms": ["करो", "जाओ", "खाओ", "कीजिए", "जाइए", "बैठिए"],
        "expected_suffixes": ["ो", "ओ", "इए", "िए"],
    },
    "Compound Verbs (संयुक्त क्रिया)": {
        "forms": ["कर लिया", "खा लिया", "जा सकता", "कर दिया", "लिख दिया", "पढ़ सकता"],
        "expected_suffixes": ["लिया", "दिया", "सकता"],
    },
    "Completive Aspect": {
        "forms": ["हो चुका", "खा चुका", "जा चुका", "कर चुकी", "पढ़ चुके"],
        "expected_suffixes": ["चुका", "चुकी", "चुके"],
    },
}

# Noun morphology and case markers
NOUN_MORPHOLOGY = {
    "Simple Postpositions (सरल परसर्ग)": {
        "examples": [
            ("राम का घर", "का"),
            ("सीता की किताब", "की"),
            ("बच्चों के खिलौने", "के"),
            ("मुझे पानी दो", "को implied"),
            ("दिल्ली से मुंबई", "से"),
            ("घर में बैठो", "में"),
            ("मेज पर रखो", "पर"),
        ],
    },
    "Compound Postpositions (संयुक्त परसर्ग)": {
        "examples": [
            ("मेरे लिए", "के लिए"),
            ("उसके साथ", "के साथ"),
            ("इसके बारे में", "के बारे में"),
            ("उसके बाद", "के बाद"),
            ("इससे पहले", "से पहले"),
            ("घर के अंदर", "के अंदर"),
            ("मेज के ऊपर", "के ऊपर"),
            ("पेड़ के नीचे", "के नीचे"),
        ],
    },
    "Plural Markers (बहुवचन)": {
        "examples": [
            ("लड़का → लड़के", "े"),
            ("लड़की → लड़कियाँ", "ियाँ"),
            ("किताब → किताबें", "ें"),
            ("कमरा → कमरे", "े"),
            ("गाड़ी → गाड़ियाँ", "ियाँ"),
        ],
    },
    "Abstract Noun Suffixes (भाववाचक प्रत्यय)": {
        "examples": [
            ("मनुष्यता", "ता"),
            ("बचपन", "पन"),
            ("लड़कपन", "पन"),
            ("मिठास", "आस"),
            ("गर्मी", "ी"),
            ("बुराई", "आई"),
            ("मनुष्यत्व", "त्व"),
        ],
    },
    "Agent Suffixes (कर्तृवाचक प्रत्यय)": {
        "examples": [
            ("दूधवाला", "वाला"),
            ("सब्जीवाली", "वाली"),
            ("पानीवाले", "वाले"),
            ("चित्रकार", "कार"),
            ("दुकानदार", "दार"),
        ],
    },
}

# Pronouns
PRONOUNS = {
    "First Person (उत्तम पुरुष)": {
        "singular": ["मैं", "मुझे", "मुझको", "मेरा", "मेरी", "मेरे"],
        "plural": ["हम", "हमें", "हमको", "हमारा", "हमारी", "हमारे"],
    },
    "Second Person (मध्यम पुरुष)": {
        "informal": ["तू", "तुझे", "तेरा", "तेरी", "तेरे"],
        "familiar": ["तुम", "तुम्हें", "तुम्हारा", "तुम्हारी", "तुम्हारे"],
        "formal": ["आप", "आपको", "आपका", "आपकी", "आपके"],
    },
    "Third Person (अन्य पुरुष)": {
        "proximal": ["यह", "इसे", "इसका", "इसकी", "इसके", "इसमें"],
        "distal": ["वह", "उसे", "उसका", "उसकी", "उसके", "उसमें"],
        "proximal_plural": ["ये", "इन्हें", "इनका", "इनकी", "इनके"],
        "distal_plural": ["वे", "उन्हें", "उनका", "उनकी", "उनके"],
    },
    "Interrogative (प्रश्नवाचक)": {
        "forms": ["कौन", "क्या", "कहाँ", "कब", "कैसे", "क्यों", "कितना", "कितनी", "कितने"],
    },
    "Relative (संबंधवाचक)": {
        "forms": ["जो", "जिसे", "जिसका", "जिसकी", "जिसके", "जिसमें"],
    },
    "Reflexive (निजवाचक)": {
        "forms": ["अपना", "अपनी", "अपने", "खुद", "स्वयं"],
    },
}

# Numbers
NUMBERS = {
    "Cardinals (गणना संख्या)": [
        "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
        "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
        "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस",
        "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ",
    ],
    "Large Numbers (बड़ी संख्याएँ)": [
        "हज़ार", "लाख", "करोड़", "अरब", "खरब",
    ],
    "Ordinals (क्रमवाचक)": [
        "पहला", "दूसरा", "तीसरा", "चौथा", "पाँचवाँ",
        "पहली", "दूसरी", "तीसरी", "चौथी", "पाँचवीं",
        "पहले", "दूसरे", "तीसरे", "चौथे", "पाँचवें",
    ],
}

# Prefixes
PREFIXES = {
    "Sanskrit Prefixes (संस्कृत उपसर्ग)": {
        "अ (negation)": ["अज्ञान", "अधर्म", "असत्य", "अशांति"],
        "अन (negation before vowel)": ["अनंत", "अनादि", "अनुचित"],
        "अधि (over/above)": ["अधिकार", "अधिपति", "अध्यक्ष"],
        "अनु (after/following)": ["अनुसार", "अनुभव", "अनुवाद"],
        "अप (away/down)": ["अपमान", "अपकार", "अपशब्द"],
        "अभि (towards)": ["अभिमान", "अभिनय", "अभिलाषा"],
        "उप (sub/near)": ["उपकार", "उपदेश", "उपयोग"],
        "दुर/दुस (bad/difficult)": ["दुर्गम", "दुर्भाग्य", "दुष्कर"],
        "निर/निस (without)": ["निर्दोष", "निर्मल", "निस्संदेह"],
        "परि (around)": ["परिवार", "परिचय", "परिणाम"],
        "प्र (forward)": ["प्रगति", "प्रकाश", "प्रयोग"],
        "प्रति (against/each)": ["प्रतिदिन", "प्रतिक्रिया", "प्रतिनिधि"],
        "वि (apart/special)": ["विशेष", "विकास", "विचार"],
        "सं/सम (together)": ["संगम", "संयोग", "समाज"],
        "सु (good/well)": ["सुंदर", "सुविधा", "सुरक्षा"],
    },
}

# Suffixes
SUFFIXES = {
    "Adjective-forming (विशेषण बनाने वाले)": {
        "ईय": ["भारतीय", "राष्ट्रीय", "जातीय"],
        "इक": ["वैज्ञानिक", "सामाजिक", "ऐतिहासिक"],
        "मय": ["आनंदमय", "दुखमय", "जलमय"],
        "पूर्ण": ["सफलतापूर्ण", "आशापूर्ण", "प्रेमपूर्ण"],
        "हीन": ["धनहीन", "बुद्धिहीन", "आशाहीन"],
        "युक्त": ["बुद्धियुक्त", "अर्थयुक्त", "शक्तियुक्त"],
        "वान/वती": ["बलवान", "धनवान", "गुणवती"],
        "शील": ["दयाशील", "कर्मशील", "परिश्रमशील"],
    },
    "Noun-forming (संज्ञा बनाने वाले)": {
        "ता": ["सुंदरता", "महानता", "अच्छाई→अच्छता"],
        "त्व": ["मनुष्यत्व", "देवत्व", "नेतृत्व"],
        "पन/पना": ["बचपन", "लड़कपन", "अपनापन"],
        "आई": ["लड़ाई", "पढ़ाई", "कमाई"],
        "आव/आवट": ["बनावट", "मिलावट", "रुकावट"],
        "आहट": ["घबराहट", "चिल्लाहट", "मुस्कराहट"],
    },
}

# Connectors and particles
CONNECTORS = {
    "Coordinating (समुच्चयबोधक)": ["और", "तथा", "एवं", "या", "अथवा", "किंतु", "परंतु", "लेकिन", "मगर"],
    "Subordinating (आश्रित)": ["कि", "जो", "जब", "तब", "यदि", "अगर", "तो", "क्योंकि", "इसलिए"],
    "Particles (निपात)": ["भी", "ही", "तो", "तक", "मात्र", "केवल", "सिर्फ"],
    "Negation (नकारात्मक)": ["न", "ना", "नहीं", "मत"],
    "Emphatic (बलार्थक)": ["जरूर", "अवश्य", "ज़रूर", "बिल्कुल"],
}

# Common words that should be single tokens
COMMON_WORDS = [
    # High frequency
    "है", "हैं", "था", "थी", "थे", "हो", "होना", "होता", "होती",
    "और", "का", "की", "के", "को", "से", "में", "पर", "तक",
    "एक", "यह", "वह", "इस", "उस", "जो", "कि", "भी", "ही",
    "नहीं", "कर", "हुआ", "कुछ", "सब", "अब", "तो", "पहले",
    # Common nouns
    "भारत", "देश", "सरकार", "लोग", "समय", "काम", "बात", "दिन", "साल",
    "घर", "पानी", "जगह", "तरह", "हिस्सा", "जीवन", "दुनिया",
    # Common verbs
    "करना", "होना", "जाना", "आना", "देना", "लेना", "कहना", "देखना", "मिलना",
]

# Test sentences for various linguistic phenomena
TEST_SENTENCES = {
    "Simple Hindi": [
        "भारत एक महान देश है।",
        "मेरा नाम राहुल है।",
        "आज मौसम बहुत अच्छा है।",
        "वह स्कूल जाता है।",
    ],
    "Complex Hindi": [
        "भारतीय संविधान विश्व का सबसे बड़ा लिखित संविधान है।",
        "विज्ञान और प्रौद्योगिकी ने मानव जीवन को बदल दिया है।",
        "स्वतंत्रता संग्राम में अनेक वीरों ने अपने प्राणों की आहुति दी।",
    ],
    "Hinglish": [
        "मुझे Python सीखना है।",
        "यह software बहुत अच्छा है।",
        "आज meeting में क्या discuss हुआ?",
        "मैंने email भेज दिया है।",
    ],
    "English": [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?",
        "Machine learning is transforming industries.",
    ],
    "Technical Hindi": [
        "कृत्रिम बुद्धिमत्ता का विकास तेज़ी से हो रहा है।",
        "संगणक विज्ञान आधुनिक युग की नींव है।",
        "इंटरनेट ने संचार को क्रांतिकारी बना दिया।",
    ],
    "Literary Hindi": [
        "जीवन एक संघर्ष है, हार मत मानो।",
        "प्रेम वह शक्ति है जो दुनिया को चलाती है।",
        "ज्ञान का दीपक अज्ञान के अंधकार को दूर करता है।",
    ],
}

# Chat template test cases
CHAT_TEST_CASES = [
    {
        "name": "Pure Hindi Conversation",
        "messages": [
            {"role": "system", "content": "आप एक मददगार AI सहायक हैं।"},
            {"role": "user", "content": "भारत की राजधानी क्या है?"},
            {"role": "assistant", "content": "भारत की राजधानी नई दिल्ली है।"},
        ],
    },
    {
        "name": "Hinglish Conversation",
        "messages": [
            {"role": "user", "content": "मुझे Python सीखना है, कहाँ से शुरू करूँ?"},
            {"role": "assistant", "content": "Python सीखने के लिए आप online courses ले सकते हैं।"},
        ],
    },
    {
        "name": "Multi-turn Hindi",
        "messages": [
            {"role": "system", "content": "आप एक हिंदी शिक्षक हैं।"},
            {"role": "user", "content": "क्रिया किसे कहते हैं?"},
            {"role": "assistant", "content": "जो शब्द किसी काम के करने या होने का बोध कराए, उसे क्रिया कहते हैं।"},
            {"role": "user", "content": "कुछ उदाहरण दीजिए।"},
            {"role": "assistant", "content": "जैसे: खाना, पीना, सोना, जाना, आना, पढ़ना, लिखना आदि।"},
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION CLASSES AND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Result of a single evaluation."""
    name: str
    passed: bool
    details: str = ""
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    expected: str = ""
    actual: str = ""


@dataclass
class EvalSummary:
    """Summary of all evaluations."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    categories: Dict[str, List[EvalResult]] = field(default_factory=dict)

    def add_result(self, category: str, result: EvalResult):
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests * 100 if self.total_tests > 0 else 0


class HindiTokenizerEvaluator:
    """Comprehensive evaluator for Hindi tokenizers."""

    def __init__(self, tokenizer_dir: str):
        self.tokenizer_dir = tokenizer_dir
        self.summary = EvalSummary()

        # Try loading HindiTokenizer wrapper (preferred - ensures consistent normalization)
        self.hindi_tokenizer = None
        try:
            from model.HindiTokenizer import HindiTokenizer
            self.hindi_tokenizer = HindiTokenizer.from_pretrained(tokenizer_dir)
            print("Using HindiTokenizer wrapper (consistent normalization)")
        except ImportError:
            print("Note: HindiTokenizer not available, using SentencePiece directly")

        # Load SentencePiece model (fallback or for direct access)
        self.sp = spm.SentencePieceProcessor()
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp.load(model_path)

        # Try loading HuggingFace tokenizer for chat template (if HindiTokenizer not available)
        self.hf_tokenizer = None
        if self.hindi_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            except Exception as e:
                print(f"Note: HuggingFace tokenizer not available ({e})")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (with normalization if HindiTokenizer available)."""
        if self.hindi_tokenizer:
            return self.hindi_tokenizer.encode(text)
        return self.sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.hindi_tokenizer:
            return self.hindi_tokenizer.decode(ids)
        return self.sp.decode(ids)

    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to token pieces (with normalization if HindiTokenizer available)."""
        if self.hindi_tokenizer:
            return self.hindi_tokenizer.tokenize(text)
        return self.sp.encode_as_pieces(text)

    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.get_piece_size()

    # ─────────────────────────────────────────────────────────────────────────
    # BASIC TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_special_tokens(self) -> List[EvalResult]:
        """Test special token IDs."""
        results = []

        special_tokens = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
        }

        for token, expected_id in special_tokens.items():
            actual_id = self.sp.piece_to_id(token)
            passed = actual_id == expected_id
            results.append(EvalResult(
                name=f"Special token: {token}",
                passed=passed,
                details=f"Expected ID {expected_id}, got {actual_id}",
                expected=str(expected_id),
                actual=str(actual_id),
            ))

        return results

    def test_roundtrip(self, texts: List[str]) -> List[EvalResult]:
        """Test encode/decode roundtrip."""
        results = []

        for text in texts:
            ids = self.encode(text)
            decoded = self.decode(ids)
            # Normalize both for comparison (SentencePiece may normalize)
            text_norm = unicodedata.normalize('NFKC', text)
            decoded_norm = unicodedata.normalize('NFKC', decoded)
            passed = decoded_norm == text_norm

            results.append(EvalResult(
                name=f"Roundtrip: {text[:40]}...",
                passed=passed,
                details=f"Tokens: {len(ids)}",
                tokens=self.encode_pieces(text),
                token_count=len(ids),
                expected=text,
                actual=decoded,
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # CONJUNCT TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_conjuncts(self) -> List[EvalResult]:
        """Test that conjuncts are preserved."""
        results = []

        for category, conjuncts in CONJUNCTS.items():
            for conj in conjuncts:
                tokens = self.encode_pieces(conj)
                # Conjunct should ideally be 1-2 tokens (with space marker)
                # It should NOT be split into consonant + halant + consonant
                token_count = len(tokens)
                # Check if halant (्) appears alone (bad splitting)
                has_isolated_halant = any('्' == t.replace('▁', '') for t in tokens)
                passed = token_count <= 2 and not has_isolated_halant

                results.append(EvalResult(
                    name=f"Conjunct ({category}): {conj}",
                    passed=passed,
                    details=f"Tokens: {tokens}",
                    tokens=tokens,
                    token_count=token_count,
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # MORPHOLOGY TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_verb_morphology(self) -> List[EvalResult]:
        """Test verb form tokenization."""
        results = []

        for category, data in VERB_MORPHOLOGY.items():
            forms = data["forms"]
            for form in forms:
                tokens = self.encode_pieces(form)
                token_count = len(tokens)
                # Verb forms should be reasonably tokenized (not over-fragmented)
                # Simple forms: 1-3 tokens, compound forms: 2-5 tokens
                is_compound = " " in form
                max_expected = 5 if is_compound else 3
                passed = token_count <= max_expected

                results.append(EvalResult(
                    name=f"Verb ({category}): {form}",
                    passed=passed,
                    details=f"Tokens: {tokens}",
                    tokens=tokens,
                    token_count=token_count,
                ))

        return results

    def test_noun_morphology(self) -> List[EvalResult]:
        """Test noun morphology and postpositions."""
        results = []

        for category, data in NOUN_MORPHOLOGY.items():
            examples = data["examples"]
            for example in examples:
                if isinstance(example, tuple):
                    text, marker = example
                else:
                    text, marker = example, ""

                tokens = self.encode_pieces(text)
                token_count = len(tokens)
                # Phrases should be reasonably tokenized
                word_count = len(text.split())
                max_expected = word_count * 3  # Allow up to 3 tokens per word average
                passed = token_count <= max_expected

                results.append(EvalResult(
                    name=f"Noun ({category}): {text}",
                    passed=passed,
                    details=f"Marker: {marker}, Tokens: {tokens}",
                    tokens=tokens,
                    token_count=token_count,
                ))

        return results

    def test_pronouns(self) -> List[EvalResult]:
        """Test pronoun tokenization."""
        results = []

        for category, subcats in PRONOUNS.items():
            for subcat, forms in subcats.items():
                for form in forms:
                    tokens = self.encode_pieces(form)
                    token_count = len(tokens)
                    # Pronouns should ideally be single tokens or 2 tokens max
                    passed = token_count <= 2

                    results.append(EvalResult(
                        name=f"Pronoun ({category}/{subcat}): {form}",
                        passed=passed,
                        details=f"Tokens: {tokens}",
                        tokens=tokens,
                        token_count=token_count,
                    ))

        return results

    def test_numbers(self) -> List[EvalResult]:
        """Test number tokenization."""
        results = []

        for category, numbers in NUMBERS.items():
            for num in numbers:
                tokens = self.encode_pieces(num)
                token_count = len(tokens)
                # Numbers should be 1-2 tokens
                passed = token_count <= 2

                results.append(EvalResult(
                    name=f"Number ({category}): {num}",
                    passed=passed,
                    details=f"Tokens: {tokens}",
                    tokens=tokens,
                    token_count=token_count,
                ))

        return results

    def test_prefixes_suffixes(self) -> List[EvalResult]:
        """Test prefix and suffix handling."""
        results = []

        # Test prefixes
        for category, prefix_data in PREFIXES.items():
            for prefix_name, examples in prefix_data.items():
                for word in examples:
                    tokens = self.encode_pieces(word)
                    token_count = len(tokens)
                    # Prefixed words should be 1-3 tokens
                    passed = token_count <= 3

                    results.append(EvalResult(
                        name=f"Prefix ({prefix_name}): {word}",
                        passed=passed,
                        details=f"Tokens: {tokens}",
                        tokens=tokens,
                        token_count=token_count,
                    ))

        # Test suffixes
        for category, suffix_data in SUFFIXES.items():
            for suffix, examples in suffix_data.items():
                for word in examples:
                    # Handle arrow notation
                    if "→" in word:
                        word = word.split("→")[0].strip()
                    tokens = self.encode_pieces(word)
                    token_count = len(tokens)
                    # Suffixed words should be 1-3 tokens
                    passed = token_count <= 3

                    results.append(EvalResult(
                        name=f"Suffix ({suffix}): {word}",
                        passed=passed,
                        details=f"Tokens: {tokens}",
                        tokens=tokens,
                        token_count=token_count,
                    ))

        return results

    def test_connectors(self) -> List[EvalResult]:
        """Test connector and particle tokenization."""
        results = []

        for category, words in CONNECTORS.items():
            for word in words:
                tokens = self.encode_pieces(word)
                token_count = len(tokens)
                # Connectors should be single tokens
                passed = token_count <= 2

                results.append(EvalResult(
                    name=f"Connector ({category}): {word}",
                    passed=passed,
                    details=f"Tokens: {tokens}",
                    tokens=tokens,
                    token_count=token_count,
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # EFFICIENCY TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_token_efficiency(self) -> Dict[str, float]:
        """Calculate token efficiency metrics."""
        metrics = {}

        # Test common words
        total_tokens = 0
        for word in COMMON_WORDS:
            tokens = self.encode_pieces(word)
            total_tokens += len(tokens)
        metrics["avg_tokens_per_common_word"] = total_tokens / len(COMMON_WORDS)

        # Test by sentence category
        for category, sentences in TEST_SENTENCES.items():
            total_chars = 0
            total_tokens = 0
            for sent in sentences:
                total_chars += len(sent)
                total_tokens += len(self.encode(sent))
            metrics[f"chars_per_token_{category}"] = total_chars / total_tokens if total_tokens > 0 else 0

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # CHAT TEMPLATE TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_chat_template(self) -> List[EvalResult]:
        """Test chat template functionality."""
        results = []

        # Use HindiTokenizer if available, otherwise HuggingFace
        tokenizer = self.hindi_tokenizer or self.hf_tokenizer
        if tokenizer is None:
            results.append(EvalResult(
                name="Chat Template",
                passed=False,
                details="No tokenizer with chat template available",
            ))
            return results

        for test_case in CHAT_TEST_CASES:
            try:
                prompt = tokenizer.apply_chat_template(
                    test_case["messages"],
                    tokenize=False
                )
                tokens = tokenizer(prompt)
                input_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
                if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]  # Handle batch dimension
                decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

                # Check roundtrip
                passed = decoded == prompt

                results.append(EvalResult(
                    name=f"Chat: {test_case['name']}",
                    passed=passed,
                    details=f"Token count: {len(input_ids)}",
                    token_count=len(input_ids),
                    expected=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    actual=decoded[:100] + "..." if len(decoded) > 100 else decoded,
                ))
            except Exception as e:
                results.append(EvalResult(
                    name=f"Chat: {test_case['name']}",
                    passed=False,
                    details=f"Error: {str(e)}",
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # EDGE CASE TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def test_edge_cases(self) -> List[EvalResult]:
        """Test edge cases."""
        results = []

        edge_cases = [
            ("Empty string", ""),
            ("Single character", "अ"),
            ("Single matra", "ा"),
            ("Punctuation", "।,;:!?"),
            ("Numbers", "१२३४५६७८९०"),
            ("Mixed script", "Hello नमस्ते 你好"),
            ("URL-like", "https://example.com"),
            ("Email-like", "test@example.com"),
            ("Long word", "अंतरराष्ट्रीयकरण"),
            ("Repeated chars", "आआआआआ"),
            ("Special Unicode", "॥ॐ॰"),
            ("Zero-width chars", "क्‍ष"),  # With ZWNJ
        ]

        for name, text in edge_cases:
            try:
                if text:
                    ids = self.encode(text)
                    decoded = self.decode(ids)
                    passed = len(ids) > 0
                else:
                    ids = self.encode(text)
                    passed = len(ids) == 0

                results.append(EvalResult(
                    name=f"Edge: {name}",
                    passed=passed,
                    details=f"Input: '{text}', Tokens: {len(ids) if text else 0}",
                    tokens=self.encode_pieces(text) if text else [],
                    token_count=len(ids) if text else 0,
                ))
            except Exception as e:
                results.append(EvalResult(
                    name=f"Edge: {name}",
                    passed=False,
                    details=f"Error: {str(e)}",
                ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # RUN ALL TESTS
    # ─────────────────────────────────────────────────────────────────────────

    def run_all_tests(self, verbose: bool = True) -> EvalSummary:
        """Run all evaluation tests."""

        print("=" * 70)
        print("HINDI TOKENIZER COMPREHENSIVE EVALUATION")
        print("हिंदी टोकननाइज़र व्यापक मूल्यांकन")
        print("=" * 70)
        print(f"Tokenizer: {self.tokenizer_dir}")
        print(f"Vocab size: {self.vocab_size()}")
        print("=" * 70)

        # Run all test categories
        test_categories = [
            ("Special Tokens", self.test_special_tokens),
            ("Conjuncts (संयुक्त अक्षर)", self.test_conjuncts),
            ("Verb Morphology (क्रिया रूप)", self.test_verb_morphology),
            ("Noun Morphology (संज्ञा रूप)", self.test_noun_morphology),
            ("Pronouns (सर्वनाम)", self.test_pronouns),
            ("Numbers (संख्याएँ)", self.test_numbers),
            ("Prefixes/Suffixes (उपसर्ग/प्रत्यय)", self.test_prefixes_suffixes),
            ("Connectors (संयोजक)", self.test_connectors),
            ("Chat Template", self.test_chat_template),
            ("Edge Cases", self.test_edge_cases),
        ]

        # Add roundtrip tests for all sentence categories
        all_sentences = []
        for sentences in TEST_SENTENCES.values():
            all_sentences.extend(sentences)

        for category_name, test_func in test_categories:
            print(f"\n{'─' * 70}")
            print(f"Testing: {category_name}")
            print('─' * 70)

            results = test_func()

            passed = sum(1 for r in results if r.passed)
            total = len(results)

            for result in results:
                self.summary.add_result(category_name, result)

                if verbose:
                    status = "✓" if result.passed else "✗"
                    print(f"  {status} {result.name}")
                    if not result.passed or verbose:
                        if result.tokens:
                            print(f"      Tokens: {result.tokens}")
                        if result.details:
                            print(f"      {result.details}")

            print(f"\n  Category Result: {passed}/{total} passed ({passed/total*100:.1f}%)")

        # Roundtrip tests
        print(f"\n{'─' * 70}")
        print("Testing: Roundtrip Encoding/Decoding")
        print('─' * 70)

        roundtrip_results = self.test_roundtrip(all_sentences)
        passed = sum(1 for r in roundtrip_results if r.passed)
        total = len(roundtrip_results)

        for result in roundtrip_results:
            self.summary.add_result("Roundtrip", result)
            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.name} ({result.token_count} tokens)")

        print(f"\n  Category Result: {passed}/{total} passed ({passed/total*100:.1f}%)")

        # Token efficiency metrics
        print(f"\n{'─' * 70}")
        print("Token Efficiency Metrics")
        print('─' * 70)

        metrics = self.test_token_efficiency()
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.2f}")

        # Final summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.summary.total_tests}")
        print(f"Passed: {self.summary.passed_tests}")
        print(f"Failed: {self.summary.failed_tests}")
        print(f"Pass Rate: {self.summary.pass_rate:.1f}%")
        print("=" * 70)

        # Category breakdown
        print("\nCategory Breakdown:")
        for category, results in self.summary.categories.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            status = "✓" if passed == total else "⚠" if passed > total * 0.8 else "✗"
            print(f"  {status} {category}: {passed}/{total} ({passed/total*100:.1f}%)")

        return self.summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Hindi Tokenizer Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="./model_hindi",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show only summary",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    evaluator = HindiTokenizerEvaluator(args.tokenizer_dir)
    summary = evaluator.run_all_tests(verbose=not args.quiet)

    if args.output:
        # Save results to JSON
        results_dict = {
            "tokenizer_dir": args.tokenizer_dir,
            "vocab_size": evaluator.vocab_size(),
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "failed_tests": summary.failed_tests,
            "pass_rate": summary.pass_rate,
            "categories": {
                cat: [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "details": r.details,
                        "tokens": r.tokens,
                        "token_count": r.token_count,
                    }
                    for r in results
                ]
                for cat, results in summary.categories.items()
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()