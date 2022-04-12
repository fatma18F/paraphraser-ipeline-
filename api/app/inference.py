# Authors: Complicated. The basis for the code was taken from internet, then it was modified to our purposes.
# Modified: Eneko Ruiz, Ruslan Mammadov
# Copyright (C) 2021 Complicated, hope DynaGroup will figure it out :)

"""
This script defines Paraphraser class that is responsible for loading the model and perform inference.
"""

import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Paraphraser:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(self.get_model_path())
        self.model = self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def get_model_path(self):
        """
        :return: Path to the paraphrasing model.
        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(this_dir, '../model')

    def inference(self, input_text):
        """
        Perform inference, i.e. generate paraphrases based on the input text.

        :param input_text: Input text which should be paraphrased
        :return: Paraphrased text
        """
        text = "paraphrase:" + input_text + " </s>"
        encoding = self.tokenizer.encode_plus(text, padding='max_length', return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=64,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=5
        )

        output_text = input_text    # Just for the case, beam_outputs should not be empty
        for beam_output in beam_outputs:
            output_text = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if output_text.lower() != input_text.lower():
                return output_text
        return output_text
