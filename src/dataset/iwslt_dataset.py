import os
import typing as t

import datasets
import torch
from datasets.utils.download_manager import DownloadManager
from torch.utils.data.dataset import Dataset

from src.utils.utils import clean_text


class Iwslt(Dataset):
    def __init__(
        self,
        version: str = "17",
        src_lan: str = "en",
        tgt_lan: str = "ro",
        data_dir: str = None,
        hugginface_tokenizer=None,
        split: str = None,
    ):
        self.version = version
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.max_length = 511

        self.dl = DownloadManager()

        self.name = f"iwslt{self.version}"

        self.version2folder = {
            "15": os.path.join(data_dir, "2015-01/texts"),
            "17": os.path.join(data_dir, "2017-01-trnted/texts"),
        }
        self.version2years = {
            "15": {"train_and_test": [2010, 2011, 2012, 2013], "dev": [2010]},
            "17": {
                "train_and_test": [2010, 2011, 2012, 2013, 2014, 2015],
                "dev": [2010],
            },
        }

        data_file = f"{self.version2folder[version]}/{src_lan}/{tgt_lan}/{src_lan}-{tgt_lan}.tgz"

        splitted_generators = self._split_generators(data_file)
        self.translation_dataset = self.load_dataset(splitted_generators, split=split)

        with torch.no_grad():
            self.tokenizer = hugginface_tokenizer

    def load_dataset(
        self,
        splitted_generators: t.List[datasets.SplitGenerator],
        split: str,
    ) -> t.List[t.Dict]:
        splitted_generators = self.concat_dataset(splitted_generators, split)

        return list(
            self._generate_examples(
                source_files=splitted_generators.gen_kwargs["source_files"],
                target_files=splitted_generators.gen_kwargs["target_files"],
            )
        )

    @staticmethod
    def concat_dataset(
        splitted_generators: t.List[datasets.SplitGenerator],
        split: str,
    ) -> datasets.SplitGenerator:
        split2ix = {"train": 0, "test": 1, "validation": 2}
        assert (
            split in split2ix
        ), "Iwslt: split must be either train or test on validation"
        if split is not None:
            return splitted_generators[split2ix[split]]

    def _split_generators(self, data_file: str) -> t.List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        pair = f"{self.src_lan}-{self.tgt_lan}"
        dl_dir = self.dl.extract(data_file)
        data_dir = os.path.join(dl_dir, f"{self.src_lan}-{self.tgt_lan}")

        years = self.version2years[self.version]["train_and_test"]
        dev = self.version2years[self.version]["dev"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"train.tags.{pair}.{self.src_lan}",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"train.tags.{pair}.{self.tgt_lan}",
                        )
                    ],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT{self.version}.TED.tst{year}.{pair}.{self.src_lan}.xml",
                        )
                        for year in years
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT{self.version}.TED.tst{year}.{pair}.{self.tgt_lan}.xml",
                        )
                        for year in years
                    ],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT{self.version}.TED.dev{year}.{pair}.{self.src_lan}.xml",
                        )
                        for year in dev
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT{self.version}.TED.dev{year}.{pair}.{self.tgt_lan}.xml",
                        )
                        for year in dev
                    ],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(
        self, source_files: t.List[str], target_files: t.List[str]
    ) -> t.List[t.Dict]:
        """Yields examples."""
        for source_file, target_file in zip(source_files, target_files):
            with open(source_file, "r", encoding="utf-8") as sf:
                with open(target_file, "r", encoding="utf-8") as tf:
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()
                            else:
                                continue

                        yield {
                            "translation": {
                                self.src_lan: source_row,
                                self.tgt_lan: target_row,
                            }
                        }

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
        return len(self.translation_dataset)

    def __getitem__(self, idx: int) -> t.Tuple[str, str]:
        sample = self.translation_dataset[idx]
        source = sample["translation"][self.src_lan]
        target = sample["translation"][self.tgt_lan]

        return source, target
