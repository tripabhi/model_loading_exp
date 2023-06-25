import time
import pathutil
import os

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class T5TranslationService:
    def __init__(self):
        self.model_type = "t5"
        self.model_dir = os.path.join(pathutil.get_model_store_path(), self.model_type)

    def fill_mask(self, model_name=None, text=None):
        if model_name is None:
            raise Exception("model_name parameter absent when calling fill_mask")
        if text is None:
            raise Exception("text parameter absent when calling fill_mask")

        model_path = os.path.join(self.model_dir, model_name)

        t_start = time.monotonic()
        tokenizer = T5Tokenizer.from_pretrained(model_path)

        t_tokenizer = time.monotonic()

        model = T5ForConditionalGeneration.from_pretrained(model_path)

        t_model = time.monotonic()

        input_ids = tokenizer(text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        t_translation = time.monotonic()

        metrics = {
            "TokenizerLoadTime": (t_tokenizer - t_start),
            "ModelLoadTime": (t_model - t_tokenizer),
            "PredictionTime": (t_translation - t_model),
        }

        return translation, metrics
