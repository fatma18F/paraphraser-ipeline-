# SCC-API

An API for Smart-Content-Creation

## Structure

The directory contains following independent modules:

* api - Code for the actual API that can run on a server
* frontend - Basic frontend for the API
* metrics - Metrics for evaluation of paraphrasing models
* notebooks - Notebooks with some relevant research we have conducted
* notes - Some relevant notes from our meetings  
* final_fine_tuning - Final training loop and relevant results

## Research

You can follow our research by looking into
* In the next "Research Overview" section
* Notes and notebooks, which contain our most relevant conclusions
* Implementation, especially but not limited by final_fine_tuning and metrics modules
* README files attached to (almost) every module

## Research Overview

### Datasets

Notebooks: [Datasets](notebooks/Datasets).

There are several datasets available for paraphrasing. You can distinguish between two kinds of datasets: 
* Datasets with source and target texts written by humans. We call them "human-made" datasets.
* Datasets with source texts written by humans and target texts obtained automatically, for example using
pivoting methods (i.e. translating text to another language and back). We call them "machine-made" datasets.
  
The "human-made" datasets have mostly good quality (i.e. fluent language, correct paraphrasers, and high dissimilarity),
but are often small. The "machine-made" datasets can be very large (even millions entries), but have worse quality 
(mostly extremely low dissimilarity, sometime incorrect paraphrasers, and even not fluent language).

##### Combined Datasets

Notebooks: [Get Combined Dataset.ipynb](/notebooks/Datasets/Get%20Combined%20Dataset.ipynb) and 
[Filter Dataset According To Dissimilarity.ipynb](/notebooks/Datasets/Filter%20Dataset%20According%20To%20Dissimilarity.ipynb).

For the final training, we joined the most promising datasets into one combined dataset. Before that, we reduced the
size of Quora dataset so that we have balanced data. Quora contains only questions, that is why we had to reduce
its size in the combined dataset.

Additionally, we created another dataset by filtering out the entries with low dissimilarity from the combined dataset 
and used both datasets for final training. 

We have also produced larger combined dataset, in which we have not reduced size of Quora. Instead, we have added ParaNMT 
dataset to create a balance between questions and non-questions. However, we were not able to use it for training because 
it took too long.

##### Details and further links
  
* Datasets we analysed with description & experiments: [Datasets](/notebooks/Datasets), 
* Datasets we used in the final training: [Get Combined Dataset.ipynb](/notebooks/Datasets/Get%20Combined%20Dataset.ipynb).
* Additional datasets that we have not analysed: [Transformers Paraphrase Data](
https://www.sbert.net/examples/training/paraphrases/README.html).

### Metrics

Notebooks: [Metrics](notebooks/Metrics)

Metrics implementation & more details: [metrics](metrics)

Good metrics are extremely important to train a paraphrasing model, because there are many ways to create a paraphrase;
and the standard metrics (like BLEU score) are not well fitted to evaluate paraphrasing quality. Therefore, 
we needed multiple metrics for different aspects of paraphrasing.

Generally, we can distinguish between 4 kinds of metrics:
1. Reference-based metrics (e.g. BLEU score): They evaluate how similar are results to the given reference.
    * Of course, in the case of paraphrases you have the problem that the output texts does not have to be similar to reference 
      in order to be valid. However, they are still helpful during training.
2. Dissimilarity metrics: They evaluate how different are paraphrases from the source texts.
3. Adequacy metrics: They evaluate the semantic similarity between source texts and paraphrases.
4. Fluency metrics: They evaluate how fluent is the output language. 
   * They are one of the most complicated metrics to evaluate, because it is hard to find robust metric for fluency. 
     We can of course easily judge the grammar, but language models normally do not have so many problems with grammar. 
     In our experience, the output of the language models normally looks very fluent and is grammatically correct. 
     It is mostly the word choice that leads to worse fluency.
   
#### Behaviour during the training:
* Dissimilarity - helpful to detect underfitting: At the beginning of training you will normally 
  have the same output as input. Then, the dissimilarity increases over epochs, which you can track with dissimilarity 
  metrics.
* Adequacy - helpful to detect overfitting: Normally, it is very high at the beginning, 
  because the model outputs the same texts as in the input.  Then, the adequacy would normally decrease very slowly, 
  which is normal, because the model learns to output more different texts. However, when the adequacy becomes very low, 
  it may mean that the model is overfitting and starts to output non-sense.
* Fluency - helpful to detect overfitting (at least in the case of BLEURT): 
  At the beginning, the fluency is very high and stable over several epochs. 
  When the models start to overfit and to output non-sense, the fluency decreases.
* Reference-based metrics - it depends on the metric, but it is mostly similar to fluency.

#### Details

* [Metrics README.md](/metrics/README.md) & [Metrics Notebooks](/notebooks/Metrics)

### Model Choice

To get a paraphrasing model, we have decided to fine-tune a language model. There are several language models
available, such as GPT-2, BART, T5 etc. We have chosen T5 models for following reasons:

* It is very popular for paraphrasing tasks, for example in Hugging Face models, and was relatively successful.
* T5 has a reputation of being very effective.
* T5 has also a reputation of the model with the best meaning capturing quality, which is important for paraphrasing.
* Differently to other language models, T5 was trained on several tasks. Therefore, it is considered to have
high generalizing ability. Btw., one of the tasks was to detect paraphrasers!
  
We have mostly concentrated on small T5 version because it is faster to train, which is important for proof-of-concept.
However, we recommend trying out larger versions of T5.

### Data Augmentation and Paraphrase Mining

Notebooks: [Machine-made Datasets](/notebooks/Datasets/Machine-made%20Datasets.ipynb)

We have not performed data augmentation, but it can be helpful to increase dataset size and the dissimilarity
between source and reference texts.

The main ideas include:
* You can switch source and target texts to double your dataset size.
* You can replace some phrases in the source texts with synonyms using [PPDB dataset](http://paraphrase.org/#/) 
  to get more samples. See:
  * [Iterative Paraphrastic Augmentation with Discriminative Span Alignment](
  https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00380/100783/Iterative-Paraphrastic-Augmentation-with)
  * [NLPAug library](https://github.com/makcedward/nlpaug)
* You can also use this approach to increase dissimilarity between source and reference texts in your dataset.
  See: 
  * [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](
  https://aclanthology.org/N19-1090/)
  * [ParaBank v2.0](https://nlp.jhu.edu/parabank/) 
* Additionally, you can use this approach as post-processing step to improve your model output.
* You can use [Paraphrase Mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)
to find more paraphrases, even for german language.
* We also recommend you to fine-tune GPT-3 model, which can be even done with only few samples. 
  Then, you can use GPT-3 model to produce more samples, and increase your dataset size with that.

### Improving Inference Speed

In general, inference should not be too slow, even with CPU. However, if you want to improve inference further, 
it by may a hard task, even maybe almost impossible if you do not use GPU. We are already using T5 small model,
and this model has always the same number of layers and weights, and it is almost the smallest version of T5. We have tried
to use batches to improve inference speed, but it has not given significant speed improvement on CPU.

So, if you want to improve inference, we recommend you to acquire GPU for your server. Additionally, you can follow
suggestions from this link to improve inference : 
[Hugging Face Transformer Inference Under 1 Millisecond Latency](
https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c). The main idea
is to convert model to the graph representation (with ONNX tool) and optimize the graph. You can use
the following [tool](https://github.com/Ki6an/fastT5) to convert T5 to ONNX. The approach will work the best
if you have GPU.

Furthermore, if you have decided to use GPU for inference, you can try to use batches to improve inference speed, which was implemented in the `batch_inference` branch.
