# Final Fine-Tuning

This directory contains script used for final fine-tuning, and most relevant results.

For results, we evaluated our model on evaluation dataset after every epoch, and you can find all produced
paraphrases and the metrics here.

You can also find plots of evaluations metrics over the epochs in the model directories.

## Structure
* t5_our_training_loop.ipynb - Our main training loop
* Results for different models - They contain metrics.json with metrics for every epoch, plots for metrics, and logs & 
predictions for every epoch
* azure_environment - Files you can use to recreate our environment. You can choose one of them based on your
preferences