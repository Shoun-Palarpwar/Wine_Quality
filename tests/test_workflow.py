import unittest

import numpy as np
import pandas as pd

from wine_quality.config import FEATURES, RAW
from wine_quality.data import load_split, validate_features
from wine_quality.evaluation import metrics, quality_predictions


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.read_csv(RAW, sep=";").drop(columns="quality").head(3)

    def test_schema_reorders_columns(self):
        actual = validate_features(self.features[self.features.columns[::-1]])
        self.assertEqual(list(actual.columns), FEATURES)

    def test_reject_invalid_inputs(self):
        invalid = [self.features.drop(columns="alcohol"), self.features.assign(extra=1),
                   self.features.assign(alcohol=np.nan), self.features.assign(alcohol=np.inf),
                   self.features.assign(alcohol=-1), self.features.assign(alcohol="invalid"),
                   self.features.iloc[:0]]
        for frame in invalid:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises((ValueError, TypeError)):
                    validate_features(frame)

    def test_ordinal_metrics(self):
        result = metrics(np.array([3, 4, 5, 6, 7, 8]), np.array([3, 5, 5, 6, 7, 6]))
        self.assertAlmostEqual(result["Accuracy"], 4 / 6)
        self.assertAlmostEqual(result["Within one point"], 5 / 6)
        self.assertAlmostEqual(result["MAE"], 0.5)

    def test_regression_rounding_and_bounds(self):
        class Model:
            def predict(self, features):
                return np.array([2.1, 5.6, 9.0])
        np.testing.assert_array_equal(quality_predictions(Model(), self.features), [3, 6, 8])

    def test_split_has_no_duplicate_rows_and_preserves_classes(self):
        train_x, train_y = load_split("train")
        test_x, test_y = load_split("test")
        train = train_x.assign(quality=train_y)
        test = test_x.assign(quality=test_y)
        self.assertEqual(len(train), 1087)
        self.assertEqual(len(test), 272)
        self.assertTrue(train.merge(test, how="inner").empty)
        self.assertEqual(set(train_y), set(test_y))


if __name__ == "__main__":
    unittest.main()
