# Authors: Ruslan Mammadov  <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and DynaGroup i.T. GmbH

"""
Tests for metrics script.

# Run it with python -m unittest test.test_metrics in metrics directory
"""

import unittest
import os
import sys

# Should not be required, but it is sometimes required
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/../src")

from src.our_metrics import Metrics

class MetricsTest(unittest.TestCase):

    def setUp(self):
        self.metrics = Metrics(bleurt_model_path=self.get_test_checkpoint_bleurt_path())

    def get_test_checkpoint_bleurt_path(self):
        """
        Return path to the BLEURT test checkpoint
        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_path = f"{this_dir}/../src/bleurt/test_checkpoint"
        if not os.path.exists(checkpoint_path):
            raise Exception("Please, update bleurt test checkpoint path in get_test_checkpoint_bleurt_path")
        return checkpoint_path

    def test_compute_metrics_with_multiple_references(self):
        self.references_batch = [['The dog bit the man.', 'The dog had bit the man.'],
                            ['It was not unexpected.', 'No one was surprised.'],
                            ['The man bit him first.', 'The man had bitten the dog.']]
        self.input_batch = self.one_reference_batch = [references[0] for references in self.references_batch]

        self.output_batch = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
        self.metrics.compute_metrics(self.input_batch, self.output_batch, self.references_batch, use_bleurt=False,
                                     use_bertscore=False)

    def perform_compute_metrics_test(self, use_bleurt=False, use_bertscore=False):
        """
        This is a smoke test. With this test, most of the code get executed.
        """
        try:
            dummy_data = self.metrics.get_dummy_data(2)
            self.metrics.compute_metrics(dummy_data, dummy_data, dummy_data, use_bleurt=use_bleurt, use_bertscore=use_bertscore)
        except Exception as e:
            self.fail(f"compute_metrics() raised {e}")

    def test_compute_metrics_without_bleurt_bert(self):
        self.perform_compute_metrics_test()

    def test_compute_metrics_with_bleurt(self):
        # Can take several minutes
        self.perform_compute_metrics_test(use_bleurt=True)

    # You should execute this test only with good internet connection, because bert will be downloaded
    def test_compute_metrics_with_bert(self):
        # Can take several minutes
        self.perform_compute_metrics_test(use_bertscore=True)


if __name__ == '__main__':
    unittest.main()
