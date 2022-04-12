# Authors: Ruslan Mammadov <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and and DynaGroup i.T. GmbH

"""
Test for process_data.py script.

Run it in api directory:
python -m unittest test.test_process_data
"""

import unittest
import app
from app.process_data import split_on_sentences, join_paraphrased_sentences


class ProcessDataTest(unittest.TestCase):

    def get_correct_splitting(self):
        """
        Get example with a text being splitted correctly.
        """
        return ["Now you’re ready for frequency distributions.",
                "A frequency distribution is essentially a table!",
                "In NLTK, frequency distributions are implemented as a distinct class called FreqDist?"]

    def get_correct_splitting_abbreviations(self):
        """
        Get example with a text with some abbreviation being splitted correctly.
        """
        return ['Mr. Brown reads magazine, esp. the good ones!',
                'He has Ph.D. in CV?',
                'Before that, he studied M.S. in Alzheimer...',
                'He is very cool, nice, etc. Nice, cool, etc., are good properties!',
                'The main difference between an M.A. and M.Sc. is the type of disciplines on which they focus.']

    def get_dummy_source(self, number):
        """
        Returns dummy source sentences.
        """
        return ["!"] * number

    def test_splitting_and_joining_normal_sentences(self):
        splitting = self.get_correct_splitting()
        text = " ".join(splitting)

        self.assertEqual(split_on_sentences(text), splitting)
        self.assertEqual(join_paraphrased_sentences(splitting, self.get_dummy_source(5)), text)

    def test_splitting_with_abbreviations(self):
        splitting = self.get_correct_splitting_abbreviations()
        text = " ".join(splitting)

        self.assertEqual(split_on_sentences(text), splitting)
        self.assertEqual(join_paraphrased_sentences(splitting, self.get_dummy_source(5)), text)

    def test_joining_invalid_paraphrasers(self):
        original_splitting = self.get_correct_splitting()
        text = " ".join(original_splitting)

        no_end_punct_splitting = [sentence[:-1] for sentence in original_splitting]
        self.assertEqual(join_paraphrased_sentences(no_end_punct_splitting, original_splitting), text)

    def test_joining_invalid_paraphrasers_and_sources(self):
        splitting = self.get_correct_splitting()
        no_end_punct_splitting = [sentence[:-1] for sentence in splitting]
        expected_output = " ".join([sentence + "." for sentence in no_end_punct_splitting])

        self.assertEqual(join_paraphrased_sentences(no_end_punct_splitting, no_end_punct_splitting), expected_output)


if __name__ == '__main__':
    unittest.main()
