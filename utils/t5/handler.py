from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5Downloader:
    def download_model_and_tokenizer(self, model_name, output_dir=None):
        if output_dir is not None:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
