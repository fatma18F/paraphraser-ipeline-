# Authors: Eneko Ruiz <eneko.ruiz@ehu.eus>
# Copyright (C) 2021 Eneko Ruiz and and DynaGroup i.T. GmbH

"""
Main class that initializes API.
"""

from typing import Optional
from process_data import split_on_sentences, join_paraphrased_sentences
from fastapi import FastAPI
from pydantic import BaseModel
from inference import Paraphraser


class ParaphraseInput(BaseModel):
    id: Optional[str] = None
    original_text: str


class ParaphraseOutput(BaseModel):
    id: Optional[str] = None
    paraphrased_text: str


# Initialize app and paraphraser. You will need internet for initialization,
# to download the tokenizer if it is not cached.
app = FastAPI()
paraphraser = Paraphraser()


@app.post('/paraphrase', response_model=ParaphraseOutput, status_code=200)
def get_paraphrase(paraphrase_input: ParaphraseInput):
    """
    POST HTTP method to get paraphrased text.

    :param paraphrase_input: Input text, in form { "original_text": "Your text." }
    :return: Paraphrased text, in form { "paraphrased_text": "Paraphrased text." }
    """
    text = paraphrase_input.original_text

    # We split the sentence, because our model was trained mainly on sentence level.
    input_sentence_list = split_on_sentences(text)
    paraphrase_sentence_list = []

    # Inference, i.e. paraphrasing
    for sentence in input_sentence_list:
        paraphrased_sentence = paraphraser.inference(sentence)
        paraphrase_sentence_list.append(paraphrased_sentence)

    # Join sentences again.
    paraphrased_text = join_paraphrased_sentences(paraphrase_sentence_list, input_sentence_list)

    return {'paraphrased_text': paraphrased_text}
