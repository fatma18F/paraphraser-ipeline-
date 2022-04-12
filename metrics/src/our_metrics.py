# Authors: Ruslan Mammadov  <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and DynaGroup i.T. GmbH

"""
Script that defines Metrics class with relevant metrics for paraphrasing.

The script includes reference-based metrics:
1. BLEU with sacrebleu tokenizer
2. BLEU with tokenizer from rouge_score library
3. Ridge scores
4. BLEURT score

Metrics for dissimilarity between input and output:
1. n-gram char overlap
2. Intersection over union
2. Diversity BLEU

Metrics for semantic similarity between input and output:
1. Bert Score

For fluency, reference-based BLEURT score can be used.
"""

import os
import sys
import pickle
import subprocess

import numpy as np

from sacrebleu import sentence_bleu, corpus_bleu
from nltk.translate import bleu_score
from datasets import load_metric
from nltk.translate import gleu_score
from argparse import Namespace
from bleurt import score

from our_tokenizers import Tokenizers
from utils import download_and_extract_zip, remove_keys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

TEMP_INPUT_FILE = f"{THIS_DIR}/cache/temp_input"
TEMP_OUTPUT_FILE = f"{THIS_DIR}/cache/temp_output"
METHOD = "method"
PARAMS = "params"

DEFAULT_BLEURT_MODEL_PATH = f"{THIS_DIR}/cache/bleurt_model"

# BLEURT models for evaluation of the model output based on reference, see https://github.com/google-research/bleurt
BleurtModelsLinks = Namespace(
    # The best model, with 579M params and 32 layers. Takes around 10 GB in RAM and CUDA.
    # Runtime for 1K inputs: no GPU - 80.4 min, with GPU - 3.8 min.
    # As you can see, GPU is required if you want to compute the metrics fast.
    # Agreement with humans - 0.228
    BLEURT_20="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",

    # Model with 167M params and 12 layers. Runtime for 1K inputs: no GPU - 24.4 min, with GPU - 1.2 min.
    # Agreement with humans - 0.219
    BLEURT_20_D12="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip",

    # Model with 45M params and 6 layers. Runtime for 1K inputs, no GPU - 5.4 min, with GPU - 0.4 min.
    # Agreement with humans - 0.211
    BLEURT_20_D6="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip",

    # Model with 30M params and 3 layers. Runtime for 1K inputs, no GPU - 2.7 min, with GPU - 0.2 min.
    # Agreement with humans - 0.191
    BLEURT_20_D3="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip"
)


class Metrics:
    def __init__(self, avoid_cache=True, python_interpreter_for_heavy_metrics=None, bleurt_model_path=None):
        """
        :param avoid_cache: If true, heavy metrics will be computed outside, so that OS can clean RAM and CUDA memory
        :param python_interpreter_for_heavy_metrics: Path to python for heavy metrics
        :param bleurt_model_path: Path to bleurt model. If None, standard path will be used
        """
        self.tokenizer = Tokenizers()
        self.nltk_bleu_smoothing_function = bleu_score.SmoothingFunction().method4
        self.avoid_cache = avoid_cache
        self.rouge = load_metric("rouge")
        self.bertscore = load_metric("bertscore")

        if avoid_cache and python_interpreter_for_heavy_metrics is None:
            self.interpreter = sys.executable
            print(f"Your python interpreter: {self.interpreter}")
            if self.interpreter is None or self.interpreter == "":
                raise Exception("Could not find python interpreter for heavy metrics."
                                "Set it explicitly using python_interpreter_for_heavy_metrics parameter (e.g. python) "
                                "or set avoid_cache to False.")

        if bleurt_model_path is None:
            if not os.path.exists(DEFAULT_BLEURT_MODEL_PATH):
                print("Warning: If you want to use bleurt model, you have to install it! "
                      "For example, you can use install_bleurt_model method.")
            self.bleurt_model_path = DEFAULT_BLEURT_MODEL_PATH
        else:
            self.set_bleurt_model_path(bleurt_model_path)

        self.bleurtscore = None

    def install_bert(self, verbose=True):
        """
        Cache bert model on hard drive.

        :param verbose: If true, output will be printed
        """
        # Bert will be installed when you call the function first time
        self.compute_bert(["Dummy example"], ["Dummy example"], verbose=verbose)

    def install_bleurt_model(self, bleurt_model_link=BleurtModelsLinks.BLEURT_20, path=DEFAULT_BLEURT_MODEL_PATH):
        """
        Installs BLEURT model from given link to the given path. You can choose different BLEURT models from
        BleurtModelsLinks.

        :param bleurt_model_link: From where to install BLEURT model
        :param path: Where to install BLEURT model, inclusive name. By default, it will be saved in cache.
        """
        print(f"Installing {bleurt_model_link} to {path}...")

        download_and_extract_zip(bleurt_model_link, path)

        self.set_bleurt_model_path(path)

    def get_dummy_data(self, num_samples=256):
        """
        Get dummy data with repeated samples to find out whether the metrics can be computed on your computer.

        :param num_samples: How many samples should be returned
        """
        return ["Dummy, dummy, dummy, very dummy but also long (a little bit) example. "
                "Just checking, does it work?"] * num_samples

    def test_bleurt(self, num_samples=256, batch_size=16):
        """
        Test on dummy data whether you computer can handle it.

        For 256 sentences, depending on model, it should take minutes/seconds on GPU.
        On CPU, for the model with less than 3 layer, it should take several minutes.
        If it takes much more time, use smaller model or do not use it!

        :param num_samples: How many samples should bleurt process
        :param batch_size: Batch size for bleurt
        """
        our_score = self.compute_bleurt(self.get_dummy_data(num_samples), self.get_dummy_data(num_samples),
                                        batch_size=batch_size)
        print(f"Test was successful! Output value for same strings but terrible english is {our_score}")

    def test_bert(self, num_samples=256):
        """
        Test on dummy data whether you computer can handle the bert evaluation.

        For 256 sentences, it should take several seconds on a good CPU, even less on GPU.
        If it takes much more time, use GPU, or use super small sets, or do not use it.
        """
        our_score = self.compute_bert(self.get_dummy_data(num_samples), self.get_dummy_data(num_samples))
        print(f"Test was successful! Output value for same strings but terrible english is {our_score}")

    def set_bleurt_model_path(self, bleurt_model_path):
        """
        Sets BLEURT model path. You do not have to call it if you installed it using install_bleurt method.
        """
        if not os.path.exists(bleurt_model_path):
            raise Exception(f"The path {bleurt_model_path} does not exist!")
        self.bleurt_model_path = bleurt_model_path
        if not self.avoid_cache:
            self.bleurtscore = score.BleurtScorer(bleurt_model_path)

    def compute_metrics(self, input_batch, output_batch, reference_batch, use_bertscore=True, use_bleurt=True,
                        verbose=True, max_samples_bertscore=None,
                        max_samples_bleurt=None, bleurt_batch_size=16):
        """
        Computes all evaluation metrics

        :param input_batch: Model's input
        :param output_batch: Model's output
        :param reference_batch: References for the outputs
        :param use_bertscore: If True, compute bert score too
        :param use_bleurt: If True, compute BLEURT score too
        :param verbose: If True, print status, warning etc.
        :param max_samples_bertscore: How many sample to evaluate for bertscore (bertscore is too slow f we have to
            many samples). If None, compute for every sample
        :param max_samples_bleurt: How many sample to evaluate for BLEURT score (BLEURT is too slow f we have to
            many samples). If None, compute for every sample
        :param bleurt_batch_size: Batch size for BLEURT model
        :return: Map from metrics to the values
        """
        if len(np.shape(reference_batch)) == 2:
            one_reference_batch = np.array(reference_batch)[:, 0]
        else:
            assert len(np.shape(reference_batch)) == 1
            one_reference_batch = reference_batch

        # Similarity to the reference
        sacre_bleu = self.compute_sacrebleu(output_batch, reference_batch)
        bleu = self.compute_bleu(output_batch, reference_batch)
        gleu = self.compute_gleu(output_batch, reference_batch)
        rouge = self.compute_rouge(output_batch, one_reference_batch, only_averages=True)

        # Not for diversity. Just interesting, for comparison to bleu between output and reference
        sacre_bleu_output_input = self.compute_sacrebleu(output_batch, input_batch)
        bleu_output_input = self.compute_bleu(output_batch, input_batch)

        # Diversity
        gleu_output_input = self.compute_gleu(output_batch, input_batch)
        bleu_diversity = self.compute_bleu_diversity(input_batch, output_batch)
        intersection_over_union = self.compute_intersection_over_union(input_batch, output_batch)
        char_ngram_overlap = self.compute_char_ngram_overlap(input_batch, output_batch)

        # Heavy metrics
        if use_bertscore:
            if max_samples_bertscore is None or max_samples_bertscore > len(reference_batch):
                max_samples_bertscore = len(reference_batch)
            bert = self.compute_bert(input_batch[:max_samples_bertscore],
                                     output_batch[:max_samples_bertscore],
                                     verbose=verbose)

        if use_bleurt:
            if max_samples_bleurt is None or max_samples_bleurt > len(reference_batch):
                max_samples_bleurt = len(reference_batch)
            bleurt = self.compute_bleurt(output_batch[:max_samples_bleurt],
                                         one_reference_batch[:max_samples_bleurt],
                                         verbose=verbose, batch_size=bleurt_batch_size)
        results = {
            "bleu": bleu,
            "sacre_bleu": sacre_bleu,
            "gleu": gleu,
            "sacre_bleu_output_input": sacre_bleu_output_input,
            "bleu_output_input": bleu_output_input,
            "gleu_output_input": gleu_output_input,
            "bleu_diversity": bleu_diversity,
            "intersection_over_union": intersection_over_union,
            "char_ngram_overlap": char_ngram_overlap,
            # 100500 metrics from rouge
            "rouge1_f1": rouge["rouge1"].fmeasure,
            "rouge1_recall": rouge["rouge1"].recall,
            "rouge1_precision": rouge["rouge1"].precision,
            "rouge2_f1": rouge["rouge2"].fmeasure,
            "rouge2_recall": rouge["rouge2"].recall,
            "rouge2_precision": rouge["rouge2"].precision,
            "rougeL_f1": rouge["rougeL"].fmeasure,
            "rougeL_recall": rouge["rougeL"].recall,
            "rougeL_precision": rouge["rougeL"].precision
        }

        if use_bertscore:
            results = {
                **results,
                "bert_f1": bert["f1"],
                "bert_precision": bert["precision"],
                "bert_recall": bert["recall"]
            }
        if use_bleurt:
            results = {
                **results,
                "bleurt": bleurt
            }
        return results

    def compute_heavy_metrics_outside(self, method, parameters, verbose=True):
        """
        Meta function that computes heavy metrics outside:

        :param method: Name of method which should be executed outside
        :param parameters: Parameters for the method
        :param verbose: If true, prints everything in the console
        :return: Results of metrics computations
        """
        task_description = {
            METHOD: method,
            PARAMS: parameters
        }
        # Write input into input file
        with open(TEMP_INPUT_FILE, "wb") as file:
            pickle.dump(task_description, file)

        # Execute script
        cmd = [self.interpreter, f"{THIS_DIR}/compute_heavy_metrics.py"]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True) as computations:
            for line in computations.stdout:
                if verbose:
                    print(line, end='')
        if verbose:
            print(f"Computed heavy metrics with return code {computations.returncode}")

        # Read output from the output file
        with open(TEMP_OUTPUT_FILE, "rb") as file:
            results = pickle.load(file)

        # Remove cache
        os.remove(TEMP_INPUT_FILE)
        os.remove(TEMP_OUTPUT_FILE)

        return results

    def compute_for_every_sample(self, func, first_argument_batch, second_argument_batch):
        """
        Meta function that computes func for every sample.

        :param func: Function that should be computed for every sample
        :param first_argument_batch: Array with samples for first argument (input/reference/output)
        :param second_argument_batch: Array with samples for second argument (input/reference/output)
        :return Arrays with values computed for every sample by func
        """
        return [func(arg1, arg2) for arg1, arg2 in zip(first_argument_batch, second_argument_batch)]

    def average_over_corpus(self, sentence_metrics_func, first_argument_batch, second_argument_batch):
        """
        Meta function that finds average for metrics over the corpus.

        :param sentence_metrics_func: Function that should be computed for every sample
        :param first_argument_batch: Array with samples for first argument (input/reference/output)
        :param second_argument_batch: Array with samples for second argument (input/reference/output)
        :return: Average metric value for the whole corpus
        """
        values = self.compute_for_every_sample(sentence_metrics_func, first_argument_batch, second_argument_batch)
        return sum(values) / len(values)

    def _get_bleu_corpus_tokenized(self, outputs, references):
        if len(np.shape(references)) == 1:
            references = [[sample] for sample in references]
        assert len(np.shape(references)) == 2

        return bleu_score.corpus_bleu(references, outputs, smoothing_function=self.nltk_bleu_smoothing_function)

    def compute_bleu(self, outputs, references):
        """
        Computes BLEU score using our tokenizer.

        :param outputs: Model's outputs
        :param references: Model's references
        :return: Bleu score for the corpus
        """
        outputs_tok, references_tok = self.tokenizer.tokenize_for_reference_based_metrics(outputs, references)
        return self._get_bleu_corpus_tokenized(outputs_tok, references_tok)

    def compute_sacreblue_for_every_sample(self, outputs, references) -> list:
        """
        Computes BLEU scores for every sample

        :param outputs: Outputs of the model
        :param references: References of the model
        :return: BLEU scores for every sample
        """
        assert len(np.shape(outputs)) == 1
        if len(np.shape(references)) == 1:
            references = np.expand_dims(references, axis=1).tolist()

        def sample_bleu(output, reference):
            return sentence_bleu(output, reference, lowercase=True, smooth_method='exp').score / 100

        return self.compute_for_every_sample(sample_bleu, outputs, references)

    def compute_sacrebleu(self, outputs, references) -> float:
        """"
        Computes Sacre-BLEU score for the corpus

        :param outputs: Outputs of the model
        :param references: References of the model
        :return: Corpus Sacre-BLEU score
        """
        assert len(np.shape(outputs)) == 1
        if len(np.shape(references)) == 1:
            references = np.expand_dims(references, axis=1).tolist()

        return corpus_bleu(list(outputs), references, lowercase=True).score / 100.

    def compute_rouge(self, outputs, references, only_averages=True):
        """
        Computes rouge score for the corpus

        :param outputs: Outputs of the model
        :param references: References of the model
        :param only_averages: If true, only the average will be saved. If False, also the max and min values.
        :return: Corpus rouge score
        """
        assert len(np.shape(outputs)) == 1
        assert len(np.shape(references)) == 1

        results = self.rouge.compute(predictions=outputs, references=references)
        if only_averages:
            for key, value in results.items():
                results[key] = value.mid
        return results

    def compute_gleu_for_every_sample(self, outputs, references, min_len=1, max_len=4):
        """
        GLEU score - Google BLEU - Computes gleu score for every sample. Computes minimum of recall and precision
        for n-grams between min_len and max_len, by default (1, 4), similarly to BLEU score.

        @see https://www.nltk.org/_modules/nltk/translate/gleu_score.html

        :param outputs: Outputs of the model
        :param references: References of the model
        :param min_len: Min n-gram size that should be taken into account
        :param maxlen: Max n-gram size that should be taken into account
        :return: List of gleu scores for every sample
        """
        assert len(np.shape(outputs)) == 1
        if len(np.shape(references)) == 1:
            references = np.expand_dims(references, axis=1).tolist()

        outputs_tok, references_tok = self.tokenizer.tokenize_for_reference_based_metrics(outputs, references)

        def sample_glue(output, reference):
            return gleu_score.sentence_gleu(references=reference, hypothesis=output,
                                            min_len=min_len, max_len=max_len)

        return self.compute_for_every_sample(sample_glue, outputs_tok, references_tok)

    def compute_gleu(self, outputs, references, min_len=1, max_len=4):
        """
        GLEU score - Google BLEU - Computes gleu score for every sample. Computes minimum of recall and precision
        for n-grams between min_len and max_len, by default (1, 4), similarly to BLEU score.

        @see https://www.nltk.org/_modules/nltk/translate/gleu_score.html

        :param outputs: Outputs of the model
        :param references: References of the model
        :param min_len: min n-gram size that should be taken into account
        :param maxlen: max n-gram size that should be taken into account
        :return: gleu score for the whoel corpus
        """
        assert len(np.shape(outputs)) == 1
        if len(np.shape(references)) == 1:
            references = np.expand_dims(references, axis=1).tolist()

        outputs_tok, references_tok = self.tokenizer.tokenize_for_reference_based_metrics(outputs, references)

        return gleu_score.corpus_gleu(list_of_references=references_tok, hypotheses=outputs_tok,
                                      min_len=min_len, max_len=max_len)

    def compute_bert_for_every_sample(self, inputs, outputs, verbose=True, use_fast_tokenizer=False, **kwargs):
        """
        Computes bert score for every sample. See https://github.com/Tiiiger/bert_score.

        :param inputs: Model's inputs
        :param outputs: Model's outputs
        :param verbose: Prints what the bertscore library outputs
        :param use_fast_tokenizer: If True, use fast tokenizers
        :param kwargs: Extra parameters, see https://github.com/Tiiiger/bert_score
        :return: list of bert score for every samples
        """
        if self.avoid_cache:
            parameters = remove_keys({**locals(), **kwargs}, ["self", "kwargs"])
            return self.compute_heavy_metrics_outside("compute_bert_for_every_sample", parameters, verbose=verbose)

        if self.bertscore is None:
            self.bertscore = load_metric("bertscore")

        results = self.bertscore.compute(predictions=outputs, references=inputs, lang="en",
                                         use_fast_tokenizer=use_fast_tokenizer, **kwargs)
        results.pop('hashcode')
        return results

    def compute_bert(self, inputs, outputs, use_fast_tokenizer=True, verbose=True, **kwargs):
        """
        Computes bert score for the corpus. See https://github.com/Tiiiger/bert_score.

        :param inputs: Model's inputs
        :param outputs: Model's outputs
        :param verbose: Prints what the bertscore library outputs
        :param use_fast_tokenizer: If True, use fast tokenizers
        :param kwargs: Extra parameters, see https://github.com/Tiiiger/bert_score
        :return: bert score for the whole dataset
        """
        results = self.compute_bert_for_every_sample(inputs, outputs, verbose=verbose,
                                                     use_fast_tokenizer=use_fast_tokenizer, **kwargs)
        for key, value in results.items():
            results[key] = np.mean(value)
        return results

    def compute_bleurt_for_every_sample(self, outputs, references, model_path=None, verbose=True, batch_size=16):
        """
        Computes BLEURT score for every sample. BLEURT is Bert model fine-tuned on human judgements on the task of
        evaluating the output based on reference. It compares the semantics similarity and the fluency of the output.

        See https://github.com/google-research/bleurt.

        :param outputs: Model's outputs
        :param references: References for given outputs
        :param model_path: Path to the BLEURT model. If not given, use the path set previously or standard path,
        :param verbose: If True, prints status, warning etc.
        :param batch_size: Batch size for BLEURT model.
        :return: List of scores for every sample
        """
        if model_path is None:
            model_path = self.bleurt_model_path

        if self.avoid_cache:
            parameters = remove_keys(locals(), ["self"])
            return self.compute_heavy_metrics_outside("compute_bleurt_for_every_sample", parameters, verbose=verbose)

        assert len(np.shape(outputs)) == 1
        assert len(np.shape(references)) == 1

        if self.bleurtscore is None:
            if not os.path.exists(model_path):
                raise Exception(
                    f"The model {model_path} does not exist. You can install the model using install_bleurt method, or"
                    f"install the model yourself and set right model_path")
            self.bleurtscore = score.BleurtScorer(model_path)

        return self.bleurtscore.score(references=references, candidates=outputs, batch_size=batch_size)

    def compute_bleurt(self, outputs, references, model_path=None, verbose=True, batch_size=16):
        """
        Computes BLEURT score for every sample. BLEURT is Bert model fine-tuned on human judgements on the task of
        evaluating the output based on reference. It compares the semantics similarity and the fluency of the output.

        See https://github.com/google-research/bleurt.

        :param outputs: Model's outputs
        :param references: References for given outputs
        :param model_path: Path to the BLEURT model. If not given, use the path set previously or standard path,
        :param verbose: If True, prints status, warning etc.
        :param batch_size: Batch size for BLEURT model.
        :return: List of scores for every sample
        """
        results = self.compute_bleurt_for_every_sample(outputs, references, model_path=model_path, verbose=verbose,
                                                       batch_size=batch_size)
        return np.mean(results)

    def sample_blue_diversity(self, input, output):
        """
        Diversity blue score

        1. Compute Blue score between output and input, not reference (!).
        2. Calculate blue in both direction and take mean
        3. We are interested in diversity => Lower Bleu = higher diversity => better results
        4. Measures "Phrasal diversity"

        :param input: Input text
        :param output: Paraphreased text
        :return Diversity bleu score
        """
        input_tok, output_tok = self.tokenizer.tokenize_for_diversity(input, output)

        def get_bleu_tokenized(text1, text2):
            return bleu_score.sentence_bleu([text1], text2, smoothing_function=self.nltk_bleu_smoothing_function)

        return (get_bleu_tokenized(input_tok, output_tok) + get_bleu_tokenized(output_tok, input_tok)) / 2

    def compute_bleu_diversity(self, input_batch, output_batch):
        """
        Computes diversity blue score for the corpus.
        @see sample_blue_diversity

        :param input_batch: Model's inputs
        :param output_batch: Model's outputs
        :return Diversity bleu score for the whole corpus
        """
        input_tok, output_tok = self.tokenizer.tokenize_for_diversity(input_batch, output_batch)

        return (self._get_bleu_corpus_tokenized(input_tok, output_tok) + self._get_bleu_corpus_tokenized(output_tok,
                                                                                                         input_tok)) / 2

    def sample_intersection_over_union(self, input, output):
        """
        ∩/∪ or Intersection/Union score
        1. Number of shared words / number of all words.
        2. Higher score = worse diversity
        3. Measures "lexical diversity"

        :param input: Input text
        :param output: Output text
        :return: Intersection over union score
        """
        input_tokenized, output_tokenized = self.tokenizer.tokenize_for_diversity(input, output)
        input_tokens = set(input_tokenized)
        output_tokens = set(output_tokenized)
        return len(input_tokens & output_tokens) / len(input_tokens | output_tokens)

    def compute_intersection_over_union(self, input_batch, output_batch):
        """
        Computes Intersection Over Union for the whole corpus. See sample_intersection_over_union

        :return: Intersection over union for the whole corpus
        """
        return self.average_over_corpus(self.sample_intersection_over_union, input_batch, output_batch)

    def sample_char_ngram_overlap(self, input, output):
        """
        Characters N-Gram overlap

        @source https://gist.github.com/avidale/dc95794227d7e53985b441ec722c4d0e

        1. It is basically the same as intersection over union, but for characters ngram in the tokenized words
        2. A little bit better than intersection over union, because it also reacts to small words changes
        3. Let's try it out just for the case

        :param input: Input text
        :param output: Output text
        :return: Characters ngram overlap for the sample
        """
        input_tok, output_tok = self.tokenizer.tokenize_for_diversity(input, output)

        def ngrams(word, n=3):
            return [word[i: i + n] for i in range(len(word) - n + 1)]

        # Extract all ngrams. Code from internet (see @source), hope it works.
        ngrams1 = {ngram for token in input_tok for n in range(3, 7) for ngram in ngrams(f' {token} ', n=n)}
        ngrams2 = {ngram for token in output_tok for n in range(3, 7) for ngram in ngrams(f' {token} ', n=n)}
        return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)

    def compute_char_ngram_overlap(self, input_batch, output_batch):
        """
        Computes characters ngram overlap the whole corpus. See sample_char_ngram_overlap

        :return: Characters ngram overlap for the whole corpus
        """
        return self.average_over_corpus(self.sample_char_ngram_overlap, input_batch, output_batch)