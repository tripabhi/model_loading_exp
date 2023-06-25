from config.constants import model_dict
import os
import pathutil

from utils.bert.handler import BertDownloader
from utils.t5.handler import T5Downloader
from utils.roberta.handler import RobertaDownloader


class Downloader:
    def __init__(self):
        self.downloaders = {
            "bert": BertDownloader(),
            "t5": T5Downloader(),
            "roberta": RobertaDownloader(),
        }

    def download_all_models(self):
        for category, models in model_dict.items():
            for _, model_name in enumerate(models):
                output_dir = os.path.join(
                    pathutil.get_model_store_path(), category, model_name
                )
                if self.downloaders[category] is not None:
                    print("Downloading model " + model_name)
                    self.downloaders[category].download_model_and_tokenizer(
                        model_name, output_dir
                    )
                else:
                    raise Exception("No downloader present for model type: " + category)


Downloader().download_all_models()
