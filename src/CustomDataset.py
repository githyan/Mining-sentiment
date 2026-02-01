import pandas as pd 
import torch
from torch.utils.data import Dataset

class CustomSentimentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=128) -> None:
        super(CustomSentimentDataset, self).__init__()
        self.texts = texts.reset_index(drop=True)
        # self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        # self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        texts = self.texts[idx]
        labels = self.labels[idx]
        
        # Input Formattings: BERT generates [SEP] and [CLS] tokens.
        #[SEP]: Marking the end of sentences, or the separation between of two sentences.
        #[CLS]: This tokens is the Beginning of the texts in sequences for classification feature tasks.
        encoding = self.tokenizer(
                texts,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
        )

        return (encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), labels)
