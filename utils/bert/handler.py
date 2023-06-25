from transformers import BertTokenizer, BertForMaskedLM


class BertDownloader:
    def download_model_and_tokenizer(self, model_name, output_dir=None):
        if output_dir is not None:
            model = BertForMaskedLM.from_pretrained(model_name)
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
