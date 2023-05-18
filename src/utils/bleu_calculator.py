import os

import datasets
import pandas as pd
from tabulate import tabulate


class BleuValues:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class BleuEvaluator(object):
    def __init__(self):
        self.metric = datasets.load_metric("sacrebleu")

    def add_element(self, model_predictions, gold_references):
        self.metric.add(predictions=model_predictions, references=gold_references)

    def add_batch(self, predictions, references):
        self.metric.add_batch(predictions=predictions, references=references)

    def final_score(self, model_predictions, gold_references):
        return self.metric.compute(predictions=model_predictions, references=gold_references)


class BleuCalculator:
    def __init__(
        self,
        dataset,
        result_dir,
    ):
        self.dataset = dataset
        self.result_dir = result_dir

    @staticmethod
    def read_csv(path_csv):
        csv_reader = pd.read_csv(path_csv, sep="\t", header=0)
        return {
            k: v
            for k, v in zip(
                csv_reader["#sentence"].tolist(), csv_reader["times"].tolist()
            )
        }

    def _retrieve_files(self):
        file2data = dict()
        for root, dirs, files in os.walk(self.result_dir):
            if any(map(lambda x: "trans" in x, files)) and "initrans" not in root:
                trans_files_name = list(filter(lambda x: ("trans" in x), files))[0]
                data = self.read_csv(path_csv=os.path.join(root, trans_files_name))
                file2data.update({trans_files_name.split(".")[-2]: data})
        return file2data

    def _load_dataset(self):
        return {i: x[1] for i, x in enumerate(self.dataset)}

    @staticmethod
    def _match_indices(method, gold):
        new_gold = dict()
        for k in gold:
            if k in method:
                new_gold.update({k: gold[k]})

        return new_gold

    @staticmethod
    def _bleu_score_formatter(bleu_score):

        bleu_dict = {
            "score": bleu_score["score"],
            "counts": str(bleu_score["counts"]),
            "totals": str(bleu_score["totals"]),
            "precisions": str(["%.2f" % prec for prec in bleu_score["precisions"]]),
            "bp": bleu_score["bp"],
            "sys_len": bleu_score["sys_len"],
            "ref_len": bleu_score["ref_len"],
        }

        return BleuValues(**bleu_dict)

    def write_report(self, file2score):
        print("Writing report...")

        # Table for the Bleu score
        header = ["Metrics"] + [m[0] for m in file2score]

        bleu_table = tabulate(
            [
                ["Score"] + [b[1].score for b in file2score],
                ["Counts"] + [b[1].counts for b in file2score],
                ["Totals"] + [b[1].totals for b in file2score],
                ["Precisions"] + [b[1].precisions for b in file2score],
                ["Bp"] + [b[1].bp for b in file2score],
                ["Sys_len"] + [b[1].sys_len for b in file2score],
                ["ref_len"] + [b[1].ref_len for b in file2score],
            ],
            headers=header,
            tablefmt="rst",
        )

        with open(os.path.join(self.result_dir, "bleu_report.txt"), mode="w") as report:
            report.write(f"Bleu Score\n{bleu_table}\n\n")

    def _compute_bleu_score(self, name, translations, gold):
        scorer = BleuEvaluator()

        translations = list(translations.values())
        gold = list(gold.values())

        gold = [[g] for g in gold]

        # for t, g in zip(translations, gold):
        #     scorer.add_element(t, [g])

        score_value = scorer.final_score(translations, gold)
        return name, self._bleu_score_formatter(score_value)

    def compute_bleu_score(self):
        file2data = self._retrieve_files()
        gold = self._load_dataset()

        if "trans_beam" in file2data:
            beam_search = self._match_indices(
                file2data["trans_gs_jacobi"], file2data["trans_beam"]
            )
            file2data["trans_beam"] = beam_search
        gold = self._match_indices(file2data["trans_gs_jacobi"], gold)

        file2score = [
            self._compute_bleu_score(file, file2data[file], gold) for file in file2data
        ]

        self.write_report(file2score)
