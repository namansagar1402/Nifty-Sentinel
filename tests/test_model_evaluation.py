import unittest

from ml_engine.classifier import TRAINING_SEED
from ml_engine.evaluation import EVALUATION_SEED, evaluate_models


class TestModelEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_models(test_samples=1000, cv_folds=3)

    def test_holdout_comes_from_an_unseen_draw(self):
        self.assertNotEqual(EVALUATION_SEED, TRAINING_SEED)
        self.assertEqual(self.report["test_size"], 1000)

    def test_reports_both_models_with_valid_metrics(self):
        for name in ("random_forest", "logistic_regression"):
            m = self.report["models"][name]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean", "cv_auc_std"):
                with self.subTest(model=name, metric=key):
                    self.assertGreaterEqual(m[key], 0.0)
                    self.assertLessEqual(m[key], 1.0)

    def test_confusion_matrix_covers_every_test_trade(self):
        cm = self.report["models"]["random_forest"]["confusion_matrix"]
        self.assertEqual(sum(sum(row) for row in cm), 1000)

    def test_roc_curve_runs_from_origin_to_corner(self):
        roc = self.report["models"]["random_forest"]["roc_curve"]
        self.assertEqual((roc["fpr"][0], roc["tpr"][0]), (0.0, 0.0))
        self.assertEqual((roc["fpr"][-1], roc["tpr"][-1]), (1.0, 1.0))

    def test_model_beats_always_say_no_baseline(self):
        rf = self.report["models"]["random_forest"]
        self.assertGreater(rf["accuracy"], self.report["majority_baseline_accuracy"])
        self.assertGreater(rf["roc_auc"], 0.5)

    def test_is_deterministic(self):
        again = evaluate_models(test_samples=1000, cv_folds=3)
        self.assertEqual(again["models"]["random_forest"]["roc_auc"],
                         self.report["models"]["random_forest"]["roc_auc"])


if __name__ == "__main__":
    unittest.main()
