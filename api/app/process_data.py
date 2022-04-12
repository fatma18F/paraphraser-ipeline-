# Authors: Ruslan Mammadov <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and and DynaGroup i.T. GmbH

"""
This script contains functions required for processing data, i.e. splitting text into sentences and back.
"""

from sentence_splitter import split_text_into_sentences


def split_on_sentences(text) -> list:
    """
    Splits text into sentences.

    :param text: Text that should be splitted
    :return: Sentences from the text as list
    """
    return split_text_into_sentences(text=text, language="en")


def join_paraphrased_sentences(paraphrased_sentences: list, source_sentences: list) -> str:
    """
    Joins paraphrased sentences into a single string. Before that, checks if there is valid punctuation at the end.
    If not, appends the valid punctuation from the original text or ".".

    :param paraphrased_sentences: Paraphrased sentences that should be fixed and joined
    :param source_sentences: Source texts. It is need for the case when the paraphrased sentence does not end with
        valid punctuation
    :return: Paraphrased sentences joined into one text
    """
    valid_sentence_ends = "!()-./\\:;<=>?[]{}"
    processed_output_sentences = []

    for source, output in zip(source_sentences, paraphrased_sentences):
        source, output = source.strip(), output.strip()  # Remove spaces

        # Check if sentence ends with valid punctuation...
        if not output[-1] in valid_sentence_ends:
            # If not, fix it.
            output += source[-1] if source[-1] in valid_sentence_ends else "."
        # Add processed output sentence to the list
        processed_output_sentences.append(output)

    return " ".join(processed_output_sentences)
