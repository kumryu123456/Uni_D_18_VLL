"""
Unit tests for Vocabulary class.
"""

import pytest
from src.data.vocab import Vocab, simple_tokenize


class TestSimpleTokenize:
    """Tests for simple_tokenize function."""

    def test_basic_tokenization(self):
        """Test basic whitespace tokenization."""
        text = "매출 추이를 보여주는 그래프"
        tokens = simple_tokenize(text)
        assert tokens == ["매출", "추이를", "보여주는", "그래프"]

    def test_punctuation_splitting(self):
        """Test punctuation is split."""
        text = "2024년 매출액은 얼마인가?"
        tokens = simple_tokenize(text)
        # Question mark should be removed/split
        assert "?" not in "".join(tokens)

    def test_empty_string(self):
        """Test empty string returns empty list."""
        assert simple_tokenize("") == []

    def test_none_input(self):
        """Test None input returns empty list."""
        assert simple_tokenize(None) == []


class TestVocab:
    """Tests for Vocab class."""

    def test_init(self):
        """Test vocabulary initialization."""
        vocab = Vocab(min_freq=2)
        assert len(vocab) == 2  # <pad>, <unk>
        assert vocab.itos == ["<pad>", "<unk>"]
        assert vocab.stoi == {"<pad>": 0, "<unk>": 1}

    def test_build_vocabulary(self):
        """Test building vocabulary from texts."""
        texts = [
            "매출 추이 그래프",
            "매출 증가 표",
            "이익 추이 차트",
        ]
        vocab = Vocab(min_freq=1)
        vocab.build(texts)

        # Check special tokens
        assert "<pad>" in vocab.stoi
        assert "<unk>" in vocab.stoi

        # Check content tokens
        assert "매출" in vocab.stoi  # Appears 2 times
        assert "추이" in vocab.stoi  # Appears 2 times

    def test_build_with_min_freq(self):
        """Test min_freq filtering."""
        texts = ["word1 word2", "word1 word3", "word1"]
        vocab = Vocab(min_freq=2)
        vocab.build(texts)

        # word1 appears 3 times (>= 2)
        assert "word1" in vocab.stoi

        # word2, word3 appear 1 time each (< 2)
        assert "word2" not in vocab.stoi
        assert "word3" not in vocab.stoi

    def test_encode_basic(self):
        """Test encoding text to IDs."""
        vocab = Vocab(min_freq=1)
        vocab.build(["hello world", "hello"])

        ids = vocab.encode("hello world")
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)

    def test_encode_unknown_token(self):
        """Test encoding unknown tokens."""
        vocab = Vocab(min_freq=1)
        vocab.build(["known token"])

        ids = vocab.encode("unknown")
        # Should return <unk> token ID (1)
        assert ids == [1]

    def test_encode_max_len(self):
        """Test max_len parameter."""
        vocab = Vocab(min_freq=1)
        vocab.build(["a b c d e f"])

        ids = vocab.encode("a b c d e f", max_len=3)
        assert len(ids) == 3

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        vocab = Vocab(min_freq=1)
        vocab.build(["some text"])

        ids = vocab.encode("")
        # Should return at least <unk>
        assert len(ids) >= 1

    def test_decode(self):
        """Test decoding IDs back to text."""
        vocab = Vocab(min_freq=1)
        vocab.build(["hello world"])

        text = "hello world"
        ids = vocab.encode(text)
        decoded = vocab.decode(ids)

        # Should recover tokens (may have spacing differences)
        assert "hello" in decoded
        assert "world" in decoded

    def test_len(self):
        """Test __len__ returns vocab size."""
        vocab = Vocab(min_freq=1)
        assert len(vocab) == 2  # Just special tokens

        vocab.build(["a b c"])
        assert len(vocab) > 2  # Special tokens + content tokens

    def test_roundtrip_encode_decode(self):
        """Test encode-decode roundtrip."""
        texts = ["매출 추이", "그래프 표시"]
        vocab = Vocab(min_freq=1)
        vocab.build(texts)

        original = "매출 추이"
        ids = vocab.encode(original)
        decoded = vocab.decode(ids)

        # Tokens should be preserved
        assert "매출" in decoded
        assert "추이" in decoded
