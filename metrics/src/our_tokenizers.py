# Authors: Ruslan Mammadov  <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and DynaGroup i.T. GmbH

"""
Script that defines Tokenizer class with tokenizers required for metrics.
"""

import numpy as np

from rouge_score import tokenize


class Tokenizers:
    def tokenize_text_for_diversity(self, text):
        """
        Tokenize text for diversity computations, i.e. Diversity bleu, n-gram overlap, etc.
        :param text: Text that should be tokenized
        :return: Tokenized text
        """
        # Remove bad punctuation
        translation_table = self._get_translation_table_for_diversity_tokenizer()
        text = text.translate(translation_table)
        return text.lower().strip().split()

    def _get_translation_table_for_diversity_tokenizer(self):
        """
        Internal function used to compute translation table for the diversity tokenizer.

        :return: Translation table, i.e. "map" from symbols to replace -> replacements
        """
        # Punctuations according to python
        punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        # Good symbols that should not be removed
        good_symbols = '\'`-'

        # Bad punctuation that should be removed
        bad_punctuations = punctuations.translate("".maketrans("", "", good_symbols))

        return "".maketrans(bad_punctuations, ' ' * len(bad_punctuations))

    def tokenize_for_diversity(self, input, output):
        """
        Tokenize the input and output for diversity. @see tokenize_text_for_diversity
        As input accepts both arrays and plain texts.
        """
        assert (type(input) == type(output))

        if type(input) == str:
            return self.tokenize_text_for_diversity(input), self.tokenize_text_for_diversity(output)
        else:
            # Input and output are batches
            input_batch = [self.tokenize_text_for_diversity(text) for text in input]
            output_batch = [self.tokenize_text_for_diversity(text) for text in output]
            return input_batch, output_batch

    def tokenize_for_reference_based_metrics(self, outputs, references):
        """
        Tokenizes the texts for reference based metrics such as gleu & rouge score (bleu uses own tokenizer).

        :param outputs: Outputs of the model
        :param references: References of the model
        :return: Tuple of tokenized outputs and tokenized references
        """
        outputs_tok = [str(tokenize.tokenize(sample, stemmer=None)) for sample in outputs]
        if len(np.shape(references)) == 2:
            references_tok = [[str(tokenize.tokenize(reference, stemmer=None)) for reference in sample] for sample in
                              references]
        elif len(np.shape(references)) == 1:
            references_tok = [str(tokenize.tokenize(sample, stemmer=None)) for sample in references]
        else:
            raise Exception(f"Dim of references should be 1 or 2, not {len(np.shape(references))}!")

        return outputs_tok, references_tok
