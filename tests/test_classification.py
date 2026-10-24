import unittest

from classification import normalize_classification_scores


class ClassificationTests(unittest.TestCase):
    def test_maps_deepfake_and_real_one_labels(self):
        scores = normalize_classification_scores(
            [
                {"label": "Deepfake", "score": 0.82},
                {"label": "Real one", "score": 0.18},
            ]
        )

        self.assertEqual(scores, {"fake": 82.0, "real": 18.0})

    def test_maps_case_insensitive_labels(self):
        scores = normalize_classification_scores(
            [
                {"label": "FAKE", "score": 0.25},
                {"label": "REAL", "score": 0.75},
            ]
        )

        self.assertEqual(scores, {"fake": 25.0, "real": 75.0})

    def test_rejects_unrecognized_model_labels(self):
        with self.assertRaisesRegex(ValueError, "classes de prédiction non reconnues"):
            normalize_classification_scores([{"label": "class_0", "score": 0.9}])


if __name__ == "__main__":
    unittest.main()
