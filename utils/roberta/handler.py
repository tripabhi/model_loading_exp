from transformers import RobertaTokenizer, RobertaForMaskedLM


class RobertaDownloader:
    def download_model_and_tokenizer(self, model_name, output_dir=None):
        if output_dir is not None:
            model = RobertaForMaskedLM.from_pretrained(model_name)
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
