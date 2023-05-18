import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

from src.utils.utils import clean_text


class Ittb(Dataset):
    def __init__(
        self,
        src_lan,
        tgt_lan,
        hugginface_tokenizer=None,
        split: str = None,
    ):
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.name = "ittb"
        self.max_length = 511

        assert (
            src_lan == "en" or src_lan == "hi"
        ), "Ittb: src_lan must be either en or hi"
        assert (
            tgt_lan == "en" or tgt_lan == "hi"
        ), "Ittb: tgt_lan must be either en or hi"
        assert src_lan != tgt_lan, "Ittb: src_lan and tgt_lan cannot be the same"

        self.translation_dataset = load_dataset("cfilt/iitb-english-hindi", split=split)

        with torch.no_grad():
            self.tokenizer = hugginface_tokenizer

    def collate_fn(self, batch):

        batch_source = [b[0] for b in batch]
        batch_target = [b[1] for b in batch]

        encoded_source = self.tokenizer(
            batch_source,
            padding=True,
            return_tensors="pt",
        )
        encoded_target = self.tokenizer(
            batch_target,
            padding=True,
            return_tensors="pt",
        )

        return {
            "source": {
                "input_ids": encoded_source["input_ids"],
                "attention_mask": encoded_source["attention_mask"],
                "sentences": batch_source,
            },
            "target": {
                "input_ids": encoded_target["input_ids"],
                "attention_mask": encoded_target["attention_mask"],
                "sentences": batch_target,
            },
        }

    def __len__(self):
        return len(self.translation_dataset)

    def __getitem__(self, idx: int):
        source = self.translation_dataset["translation"][idx][self.src_lan]
        target = self.translation_dataset["translation"][idx][self.tgt_lan]

        return source, target
