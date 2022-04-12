# Metrics

This module implements relevant metrics for paraphrasing.

## API
Main usage:
```
from our_metrics import Metrics, BleurtModelsLinks

# Setting avoid_cache to True will force heavy metrics to be computed in a separate script, 
# so that the RAM and cuda memory will be cleared by OS after finish.
metrics = Metrics(avoid_cache=True)

# If bleurt is not installed. You have to install it only once!
# You can also set path to the BLEURT model in constructor or using set_bleurt_model_path.
metrics.install_bleurt_model(bleurt_model_link=BleurtModelsLinks.BLEURT_20)

# You can also install bert, for which you need internet connection.
# Otherwise bert will be installed with the first usage (differently from BLEURT).
metrics.install_bert()

# If you are not sure whether your computer can handle bert and bleurt
metrics.test_bleurt(num_samples=256, batch_size=64)
metrics.test_bert(num_samples=256)

input = ["input1", "input2"]
output = ["output1", "output2"]
references = ["reference1", "reference2"]

results = metrics.compute_metrics(input, output, references)
```

Details:
1. It is also possible to execute every metric separately for corpus and for single sample/all samples.
2. You can use multiple references. However, some metrics may ignore other than the first reference.
3. You can install and use different versions of BLEURT. Bigger models will lead to more precise metrics, but
require more memory and are slower, see https://github.com/google-research/bleurt.
4. Look at the additional parameters you can specify before using the methods.

## Metrics used

### Reference-based metrics
1. BLEU with Sacre-BLEU tokenizer
2. BLEU with the tokenizer from rouge_score library
3. Ridge scores
4. BLEURT score

### Dissimilarity metrics
Metrics for dissimilarity between input and output:
1. n-gram char overlap - dissimilarity on character level
2. Intersection over union - dissimilarity on syntax level
2. Diversity BLEU - dissimilarity on phrase level

### Semantic similarity metrics
Metrics for "meaning preserving" between input and output (for paraphrases):
1. Bert Score

### Fluency
For fluency, reference-based BLEURT score can be used. See https://github.com/google-research/bleurt.

## Source structure
1. BLEURT - it contains code for BLEURT metrics published by "The Google AI Language Team Authors" 
   under Apache License 2
2. cache - directory for cache, such as temporarily files and models required for metrics (BLEURT model)
3. out_metrics.py - script with Metrics class which implements the metrics
4. compute_heavy_metrics.py - auxiliary script for computing heavy metrics, i.e. metrics that require fine-tuned language models
5. utils.py, our_tokenizer.py - contain utilities for metrics script

## Requirements 
You will need following libraries, ideally the latest version (as for 2021-11-26)

* numpy
* wget  
* nltk
* tensorflow
* sacrebleu
* rouge
* datasets
* rouge_score
* bert_score

## Research
You can look into our research that was documented in notebooks/Metrics and some notes.

## Possible improvements

* Use SBert to evaluate semantic similarity
* Add some model that can evaluate paraphrasers in respect to semantic similarity 
  (see example in notebooks/Metrics/Bert based fine-tuned model for paraphrasers evaluation)     
* Implement Parse Tree Distance metrics for dissimilarity. It would be helpful if you have high syntax dissimilarity 
  and want to go level up and check sentence structures
* Add more metrics for fluency, such as grammar score, perplexity
* Add more tests

## Answers to some expected questions
Question: Why do you compute heavy metrics in a separate script?

Answer: Such metrics as Bert and BLEURT require language models to be loaded to memory. I tried different ways
to remove them from the memory after finish, but they didn't work perfectly. So, we decided to execute them in separate
script so that OS can clear the memory for us.

Question: Why do you have different tokenizers for diversity and for reference-based metrics:

Answer: For diversity, we are more interested in the original tokens, to check whether they were replaced/changed.
For reference-based metrics, we want to use the same tokenizers that are used in the research.

Question: Why do you have Sacre-BLEU and BLEU with another tokenizer?

Answer: Sacre-BLEU was unstable, so we decided to introduce second BLEU score with another tokenizer.

Question: Why don't you use perplexity for fluency, but BLEURT model?

Answer: Using perplexity would also be a good idea. However, perplexity would require to have a language model that
would output probability, such as GPT-3, BERT, etc. Since we are limited in resources, we wanted to be sure that if 
we use heavy metrics, we use the robust ones. Perplexity, however, is not used widely in research. 

Probably, the reason is that the perplexity may be not robust - it depends on what samples the language model have seen 
before. You may have perfect fluent text and get bad perplexity, if the language model is not familiar with this kind 
of language or topic. In the research, fluency is often evaluated by comparison to the reference. That is why we have chosen 
BLEURT.

Question: How this or other metrics is computed, what it means?

Answer: Check documentation in code and notebooks/Metrics/Metrics.ipynb notebook.

Question: Why haven't you used this or other metric?

Answer: We were limited in time, so we may have missed some useful metrics.