"""Unit tests for feature extraction modules."""

import numpy as np
import pytest

from depression_detector.features.text_features import TfidfFeatureExtractor
from depression_detector.features.psychological_features import PsychologicalFeatureExtractor


class TestTfidfFeatureExtractor:
    def test_fit_transform_returns_matrix(self, sample_texts, tfidf_extractor):
        matrix = tfidf_extractor.fit_transform(sample_texts)
        assert matrix.shape[0] == len(sample_texts)
        assert matrix.shape[1] > 0

    def test_transform_before_fit_raises(self, sample_texts):
        extractor = TfidfFeatureExtractor(word_max_features=100, use_char_ngrams=False)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            extractor.transform(sample_texts)

    def test_feature_names_match_shape(self, sample_texts, tfidf_extractor):
        tfidf_extractor.fit(sample_texts)
        names = tfidf_extractor.get_feature_names()
        matrix = tfidf_extractor.transform(sample_texts)
        assert len(names) == matrix.shape[1]

    def test_char_ngrams_increase_feature_count(self, sample_texts):
        extractor_word_only = TfidfFeatureExtractor(
            word_max_features=500, use_char_ngrams=False
        )
        extractor_both = TfidfFeatureExtractor(
            word_max_features=500, char_max_features=200, use_char_ngrams=True
        )
        mat_word = extractor_word_only.fit_transform(sample_texts)
        mat_both = extractor_both.fit_transform(sample_texts)
        assert mat_both.shape[1] > mat_word.shape[1]

    def test_save_load_roundtrip(self, sample_texts, tmp_path):
        extractor = TfidfFeatureExtractor(word_max_features=200, use_char_ngrams=False)
        extractor.fit(sample_texts)
        mat1 = extractor.transform(sample_texts).toarray()

        save_path = tmp_path / "tfidf.pkl"
        extractor.save(save_path)

        loaded = TfidfFeatureExtractor.load(save_path)
        mat2 = loaded.transform(sample_texts).toarray()
        np.testing.assert_array_almost_equal(mat1, mat2)


class TestPsychologicalFeatureExtractor:
    def test_extract_one_returns_dict(self, psych_extractor, sample_texts):
        result = psych_extractor.extract_one(sample_texts[0])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_feature_names_present(self, psych_extractor, sample_texts):
        result = psych_extractor.extract_one(sample_texts[0])
        for name in psych_extractor.feature_names:
            assert name in result, f"Missing feature: {name}"

    def test_transform_shape(self, psych_extractor, sample_texts):
        matrix = psych_extractor.transform(sample_texts)
        assert matrix.shape == (len(sample_texts), len(psych_extractor.feature_names))

    def test_depressed_text_higher_first_person_ratio(self, psych_extractor):
        depressed = "I feel so worthless and I hate myself. I can't do anything right."
        normal = "The weather is nice. We went to the park. Everyone had fun."
        d_feats = psych_extractor.extract_one(depressed)
        n_feats = psych_extractor.extract_one(normal)
        assert d_feats["ratio_first_person_singular"] > n_feats["ratio_first_person_singular"]

    def test_depressed_text_higher_negative_emotion(self, psych_extractor):
        depressed = "I feel hopeless, miserable, and worthless every single day."
        normal = "I had a wonderful and amazing day with great friends."
        d_feats = psych_extractor.extract_one(depressed)
        n_feats = psych_extractor.extract_one(normal)
        assert d_feats["ratio_negative_emotion"] > n_feats["ratio_negative_emotion"]

    def test_empty_text_returns_zeros(self, psych_extractor):
        feats = psych_extractor.extract_one("")
        assert all(v == 0.0 for v in feats.values())

    def test_transform_as_dataframe(self, psych_extractor, sample_texts):
        df = psych_extractor.transform_as_dataframe(sample_texts)
        assert df.shape == (len(sample_texts), len(psych_extractor.feature_names))
        assert list(df.columns) == psych_extractor.feature_names
