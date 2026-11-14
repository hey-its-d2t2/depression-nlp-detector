"""Unit tests for dataset loading and splitting."""

import pytest

from depression_detector.data.dataset import DepressionDataset


class TestDepressionDataset:
    def test_synthetic_generation(self, synthetic_dataset):
        assert len(synthetic_dataset) == 200
        dist = synthetic_dataset.class_distribution()
        assert "depressed" in dist
        assert "non-depressed" in dist

    def test_synthetic_is_balanced(self, synthetic_dataset):
        dist = synthetic_dataset.class_distribution()
        assert abs(dist["depressed"] - dist["non-depressed"]) <= 10  # near-balanced

    def test_split_sizes(self, synthetic_dataset):
        train, val, test = synthetic_dataset.split(test_size=0.2, val_size=0.1)
        total = len(train) + len(val) + len(test)
        assert abs(total - len(synthetic_dataset)) <= 2  # rounding tolerance

    def test_from_dataframe(self):
        ds = DepressionDataset.from_dataframe({
            "text": ["feeling great", "feeling terrible", "happy day", "sad day"],
            "label": [0, 1, 0, 1],
        })
        assert len(ds) == 4

    def test_missing_column_raises(self):
        with pytest.raises(AssertionError):
            DepressionDataset.from_dataframe({"content": ["text"], "sentiment": [1]})

    def test_save_and_load_csv(self, synthetic_dataset, tmp_path):
        save_path = tmp_path / "dataset.csv"
        synthetic_dataset.save_csv(save_path)
        loaded = DepressionDataset.from_csv(save_path)
        assert len(loaded) == len(synthetic_dataset)

    def test_split_maintains_class_ratio(self, synthetic_dataset):
        train, val, test = synthetic_dataset.split(test_size=0.2, val_size=0.1)
        # All splits should have both classes
        assert len(set(train.labels)) == 2
        assert len(set(val.labels)) == 2
        assert len(set(test.labels)) == 2
