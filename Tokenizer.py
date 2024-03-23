# TODO: CREATE A TOKENIZER FROM SCRATCH

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        assert self.tokenizer.name_or_path == "facebook/bart-large-cnn", f"""Wrong tokenizer {self.tokenizer.name_or_path} utilized. Please use "facebook/bart-large-cnn" tokenizer."""
    
    def get_tokenizer(self):
        # print("loading the tokenizer...")
        return self.tokenizer
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
