import torch
from torch.utils.data import Dataset
import datasets

from src.utils.utils import retrieve_map_languages_flores, clean_text
import typing as t


class Flores(Dataset):
    def __init__(
        self,
        src_lan: str = "ro",
        tgt_lan: str = "en",
        hugginface_tokenizer=None,
        split: str = None,
    ):
        self.name = "flores"
        self.max_length = 511
        self.src_lan = retrieve_map_languages_flores(src_lan).lower()[:3]
        self.tgt_lan = retrieve_map_languages_flores(tgt_lan).lower()[:3]

        if "test" in split:
            split = "dev" + split

        self.translation_dataset_src = datasets.load_dataset(
            "gsarti/flores_101", self.src_lan, split=split
        )
        self.translation_dataset_tgt = datasets.load_dataset(
            "gsarti/flores_101", self.tgt_lan, split=split
        )

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

    def __len__(self) -> int:
        return self.translation_dataset_src.num_rows

    def __getitem__(self, idx: int) -> t.Tuple[str, str]:
        source = str(self.translation_dataset_src.data.column(6)[idx])
        target = str(self.translation_dataset_tgt.data.column(6)[idx])

        return source, target
