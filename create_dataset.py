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
        return self.tokenizer(example, padding="longest", max_length=1024, truncation=True, return_attention_mask=return_attention_mask) # max_length = 2500 because => 10k len article:2220 tokens; aiming for 11k as that covers around 99% of the train dataset

    def get_dataset(self) -> datasets:
        print("loading the dataset...")
        # split="train[10%:50%]+train[70%:80%]+test+validation"
        # dataset = datasets.load_dataset(self.dataset_name, self.dataset_version)
        # dataset_article = dataset.map(self.tokenization, input_columns=["article"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=self.config.batch)
        # dataset_highlights = dataset.map(self.tokenization, input_columns = ["highlights"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=self.config.batch)

        # dataset_article.set_format(type="torch")
        # dataset_highlights.set_format(type="torch")

        
        train_dataset = datasets.load_dataset("cnn_dailymail", "1.0.0", split="train[:10%]")
        validation_dataset = datasets.load_dataset("cnn_dailymail", "1.0.0", split="validation[:10%]")
        test_dataset = datasets.load_dataset("cnn_dailymail", "1.0.0", split="test[:10%]")

        train_dataset_article = train_dataset.map(self.tokenization, input_columns=["article"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)
        train_dataset_highlights = train_dataset.map(self.tokenization, input_columns = ["highlights"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)

        validation_dataset_article = validation_dataset.map(self.tokenization, input_columns=["article"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)
        validation_dataset_highlights = validation_dataset.map(self.tokenization, input_columns = ["highlights"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)

        test_dataset_article = test_dataset.map(self.tokenization, input_columns=["article"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)
        test_dataset_highlights = test_dataset.map(self.tokenization, input_columns = ["highlights"], remove_columns=["article", "highlights", "id"], batched=True, batch_size=32)

        train_dataset_article.set_format(type="torch")
        train_dataset_highlights.set_format(type="torch")

        validation_dataset_article.set_format(type="torch")
        validation_dataset_highlights.set_format(type="torch")

        test_dataset_article.set_format(type="torch")
        test_dataset_highlights.set_format(type="torch")

        return train_dataset_article, train_dataset_highlights, validation_dataset_article, validation_dataset_highlights, test_dataset_article, test_dataset_highlights