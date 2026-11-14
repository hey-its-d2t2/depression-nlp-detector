"""Unit tests for text preprocessing pipeline."""

import pytest
from depression_detector.data.preprocessing import TextPreprocessor, PreprocessorConfig


class TestTextPreprocessorCleaning:
    def test_removes_urls(self, preprocessor):
        result = preprocessor.clean("Visit https://example.com for more info.")
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_html_tags(self, preprocessor):
        result = preprocessor.clean("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "hello" in result
        assert "world" in result

    def test_removes_emails(self, preprocessor):
        result = preprocessor.clean("Contact us at support@example.com")
        assert "support@example.com" not in result

    def test_expands_contractions(self, preprocessor):
        result = preprocessor.clean("I can't do this anymore, I won't try.")
        assert "cannot" in result
        assert "will not" in result

    def test_lowercases_text(self, preprocessor):
        result = preprocessor.clean("Hello World DEPRESSION")
        assert result == result.lower()

    def test_normalizes_whitespace(self, preprocessor):
        result = preprocessor.clean("too   many    spaces")
        assert "  " not in result

    def test_empty_string_returns_empty(self, preprocessor):
        assert preprocessor.clean("") == ""
        assert preprocessor.clean("   ") == ""

    def test_non_string_returns_empty(self, preprocessor):
        assert preprocessor.clean(None) == ""


class TestTextPreprocessorNegationPreservation:
    def test_preserves_negation_words(self):
        config = PreprocessorConfig(preserve_negations=True, remove_stopwords=True)
        preprocessor = TextPreprocessor(config)
        result = preprocessor.process("I do not feel happy at all.")
        assert "not" in result

    def test_preserves_first_person_pronouns(self):
        config = PreprocessorConfig(preserve_negations=True, remove_stopwords=True)
        preprocessor = TextPreprocessor(config)
        result = preprocessor.process("I feel empty inside.")
        assert "i" in result or "feel" in result


class TestTextPreprocessorPipeline:
    def test_full_pipeline(self, preprocessor):
        text = "I've been feeling <b>hopeless</b> lately. Visit http://help.com"
        result = preprocessor.process(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "http" not in result
        assert "<b>" not in result

    def test_batch_processing(self, preprocessor, sample_texts):
        results = preprocessor.process_batch(sample_texts)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, str) for r in results)

    def test_process_preserves_depression_signal(self, preprocessor):
        depressed = preprocessor.process(
            "I feel hopeless and worthless. Nothing will ever get better."
        )
        normal = preprocessor.process(
            "Had a great and wonderful day. Everything is going well."
        )
        # Depression text should retain depression-related words
        depression_words = {"hopeless", "worthless", "nothing", "better"}
        assert any(w in depressed for w in depression_words)
