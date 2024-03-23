import datasets
from Tokenizer import Tokenizer

class TransformerDataset:
    def __init__(self, config:dict, dataset_name:str, dataset_version:str=None):
        super(TransformerDataset, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.config = config

        self.tokenizer =Tokenizer().get_tokenizer()

    def tokenization(self, example, return_attention_mask=True):
        return self.tokenizer(example, padding="longest", max_length=2500, truncation=True, return_attention_mask=return_attention_mask) # max_length = 2500 because => 10k len article:2220 tokens; aiming for 11k as that covers around 99% of the train dataset

    def get_dataset(self) -> datasets:
        print("loading the dataset...")
        # split="train[10%:50%]+train[70%:80%]+test+validation"
        dataset = datasets.load_dataset(self.dataset_name, self.dataset_version)
        dataset_article = dataset.map(self.tokenization, input_columns=["article"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=self.config.batch)
        dataset_highlights = dataset.map(self.tokenization, input_columns = ["highlights"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=self.config.batch)

        dataset_article.set_format(type="torch")
        dataset_highlights.set_format(type="torch")

        return dataset_article, dataset_highlights
