import time
import pathutil
import os

from transformers import AutoTokenizer, RobertaForMaskedLM
import torch


class RobertaFillMaskService:
    def __init__(self):
        self.model_type = "roberta"
        self.model_dir = os.path.join(pathutil.get_model_store_path(), self.model_type)

    def fill_mask(self, model_name=None, text=None):
        if model_name is None:
            raise Exception("model_name parameter absent when calling fill_mask")
        if text is None:
            raise Exception("text parameter absent when calling fill_mask")

        model_path = os.path.join(self.model_dir, model_name)

        t_start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        t_tokenizer = time.monotonic()

        model = RobertaForMaskedLM.from_pretrained(model_path)

        t_model = time.monotonic()

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        prediction = tokenizer.decode(predicted_token_id)

        t_prediction = time.monotonic()

        metrics = {
            "TokenizerLoadTime": (t_tokenizer - t_start),
            "ModelLoadTime": (t_model - t_tokenizer),
            "PredictionTime": (t_prediction - t_model),
        }

        return prediction, metrics
