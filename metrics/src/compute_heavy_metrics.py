# Authors: Ruslan Mammadov  <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and DynaGroup i.T. GmbH

"""
Auxiliary script for computing heavy metrics, i.e. metrics that require fine-tuned language models.

If RAM or CUDA memory cache should be avoided, computing these metrics in another script can
ensure that after the evaluation, the CUDA and RAM memory will be released by OS.
"""

import pickle

from our_metrics import Metrics, TEMP_INPUT_FILE, TEMP_OUTPUT_FILE, METHOD, PARAMS

if __name__ == "__main__":
    # Do not avoid cache here, OS will remove cache after script is finished.
    the_metrics = Metrics(avoid_cache=False)

    # Read the input & task description from the input file
    with open(TEMP_INPUT_FILE, "rb") as file:
        task = pickle.load(file)

    # Compute metrics
    print(f"Starting executing of {task[METHOD]}.")
    if task[METHOD] == "compute_bert_for_every_sample":
        results = the_metrics.compute_bert_for_every_sample(**task[PARAMS])
    elif task[METHOD] == "compute_bleurt_for_every_sample":
        results = the_metrics.compute_bleurt_for_every_sample(**task[PARAMS])
    else:
        results = None
        print(f"The method {task[METHOD]} is not supported!")

    # Write down results into output file
    print("Execution was succesful!")
    with open(TEMP_OUTPUT_FILE, "wb") as file:
        pickle.dump(results, file)

else:
    # This script is supposed to be executed directly, not imported as library.
    raise Exception("Error, this script should be executed, not imported!")

