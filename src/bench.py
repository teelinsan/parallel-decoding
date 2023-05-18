import csv
import logging
import os
import sys
import time

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import utils
from src.utils.bench_scorer import Scorer
from src.utils.utils import retrieve_model_name, check_zero_division

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("bench")


class MTBenchmarker(object):
    def __init__(
            self,
            dataset: Dataset,
            decoders,
            src_lang,
            tgt_lang,
            compare_to: str = "autoregressive",
            result_dir: str = None,
            device: str = "cuda",
            debug: bool = False,
    ):
        self.dataset = dataset
        self.decoders = decoders
        self.device = device
        self.debug = debug

        self.compare_to = compare_to

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = self.decoders[0].model_name.split("/")[1]

        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=False
        )

        self.result_dir = result_dir
        self.exp_dir = self._retrieve_exp_dir()

    def _synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _retrieve_exp_dir(self):
        file_name = self._retrieve_file_name()

        exp_dir = os.path.join(self.result_dir, "benchmark", file_name)
        utils.makedirs(exp_dir)

        return exp_dir

    def _retrieve_file_name(self):
        model_name = retrieve_model_name(self.model_name)
        lang = f"{self.src_lang}_{self.tgt_lang}"
        return f"{model_name}/{self.device}/{self.dataset.name}/{lang}"

    @staticmethod
    def _write_on_file(path, item, i):

        utils.makedirs(path)
        with open(path, "a") as file:
            writer = csv.writer(file, delimiter="\t")
            # write header
            if os.stat(path).st_size == 0:
                writer.writerow(["i", "item"])
            writer.writerow([i, item])

    def _compute_info_table(self, best_alg, i, grid):
        # Table for the test info
        info_table = tabulate([
                [
                    self.decoders[0].model_name,
                    self.dataset.name,
                    self.device,
                    best_alg,
                    i,
                    f"{self.src_lang}-{self.tgt_lang}",
                ]
            ],
            headers=[
                "Model",
                "Dataset",
                "Device",
                "Best Algorithm",
                "Sentences",
                "Languages",
            ],
            tablefmt=grid,
        )

        return info_table

    def _compute_time_table(self, scorers, grid):

        if len(scorers) == 1:
            name, scorer = list(scorers.items())[0]
            times_table = tabulate([
                ["Time", scorer.tot_mean_time],
                ["Iteration", scorer.tot_mean_iter],
            ],
                headers=["Metrics", name], tablefmt=grid,
            )
        else:
            tests_header = ["Metrics"] + [name for name in scorers]

            comp_scorer = scorers.get(self.compare_to)
            time_speedup = [check_zero_division(comp_scorer.tot_mean_time, scorer.tot_mean_time) for scorer in scorers.values()]
            iter_speedup = [check_zero_division(comp_scorer.tot_mean_iter, scorer.tot_mean_iter) for scorer in scorers.values()]

            times_table = tabulate(
                [
                    ["Speedup T"] + time_speedup,
                    ["Speedup I"] + iter_speedup,
                    ["Time"] + [scorer.tot_mean_time for scorer in scorers.values()],
                    ["Iter"] + [scorer.tot_mean_iter for scorer in scorers.values()],
                ],
                headers=tests_header, tablefmt=grid,
            )

        return times_table

    def _compute_bleu_table(self, scorers, grid):

        bleu_scores = [scorer.compute_bleu_score() for scorer in scorers.values()]

        # Table for the Bleu score
        bleu_table = tabulate([
            ['Score'] + [score.score for score in bleu_scores],
            ['Counts'] + [score.counts for score in bleu_scores],
            ['Totals'] + [score.totals for score in bleu_scores],
            ['Precisions'] + [score.precisions for score in bleu_scores],
            ['Bp'] + [score.bp for score in bleu_scores],
            ['Sys_len'] + [score.sys_len for score in bleu_scores],
            ['ref_len'] + [score.ref_len for score in bleu_scores],
        ], headers=["Metrics"] + [name for name in scorers], tablefmt=grid)

        return bleu_table

    def write_report(self, i, scorers, best_algorithm):
        print("Writing report...")

        # Compute best algorithm
        best_alg = max(best_algorithm, key=best_algorithm.get)
        info_table_txt = self._compute_info_table(best_alg, i, grid="grid")
        info_table_tex = self._compute_info_table(best_alg, i, grid="latex")

        # Table for the benchmark times
        times_table_txt = self._compute_time_table(scorers, grid="rst")
        times_table_tex = self._compute_time_table(scorers, grid="latex")

        # Table for the bleu score
        bleu_table_txt = self._compute_bleu_table(scorers, grid="rst")

        print(self.exp_dir)

        with open(os.path.join(self.exp_dir, "report.txt"), mode="w") as report:
            report.write(f"Test Info\n{info_table_txt}\n\n")
            report.write(f"Benchmark\n{times_table_txt}\n\n")
            report.write(f"Bleu\n{bleu_table_txt}\n\n")

        with open(os.path.join(self.exp_dir, "latex.txt"), mode="w") as report:
            report.write(f"Test Info\n{info_table_tex}\n\n")
            report.write(f"Benchmark\n{times_table_tex}\n\n")

    def write_inline(self, i: int, scorers, best_alg):

        for name, scorer in scorers.items():
            # Write times
            path = os.path.join(self.exp_dir, name, f"{name}.tsv")
            self._write_on_file(path, scorer.current_time, i)
            # Write iterations
            path = os.path.join(self.exp_dir, name, f"iter_{name}.tsv")
            self._write_on_file(path, scorer.current_iter, i)
            # Write translations
            path = os.path.join(self.exp_dir, name, f"trans_{name}.tsv")
            self._write_on_file(path, scorer.current_transl, i)
            # Write initializations
            path = os.path.join(self.exp_dir, name, f"init_{name}.tsv")
            self._write_on_file(path, scorer.current_init, i)

        # Write mean
        path = os.path.join(self.exp_dir, "meanvar.tsv")
        utils.makedirs(path)
        with open(path, "a") as file:
            writer = csv.writer(file, delimiter="\t")

            # Write header
            if os.stat(path).st_size == 0:
                header = ["#sentence"] + [f"mean_{name}" for name in scorers] + ["best_alg"]
                writer.writerow(header)

            row = [i] + [scorer.current_time for scorer in scorers.values()] + [best_alg]
            writer.writerow(row)

    @staticmethod
    def _compute_best_algorithm(scorers):

        best = scorers.get(min(scorers, key=lambda x: scorers[x].current_time)).acronym

        return best

    def _compute_postfix(self, scorers, best_algorithms, curr_alg):

        best_alg = max(best_algorithms, key=best_algorithms.get)

        if len(scorers) == 1:
            _, scorer = list(scorers.items())[0]
            postfix = {
                scorer.acronym: (
                    round(scorer.tot_mean_time, 3),
                    round(scorer.tot_mean_iter, 3)
                ),
                "ca": curr_alg,
                "ba": best_alg,
            }

        else:
            comp_scorer = scorers.get(self.compare_to)

            postfix = {
                scorer.acronym: (
                    check_zero_division(comp_scorer.tot_mean_time, scorer.tot_mean_time),
                    check_zero_division(comp_scorer.tot_mean_iter, scorer.tot_mean_iter)
                ) for name, scorer in scorers.items() if name != self.compare_to
            }

            postfix.update({"ca": curr_alg, "ba": best_alg})

        return postfix

    def compute_total_time(self):

        i = 0
        scorers = {decoder.name: Scorer(decoder.name, decoder.acronym) for decoder in self.decoders}
        best_algorithms = {decoder.acronym: 0 for decoder in self.decoders}

        pbar = tqdm(self.dataloader, desc="Computing Benchmark...")
        for x in pbar:

            try:
                input_ids = x["source"]["input_ids"].to(self.device)
                attention_mask = x["source"]["attention_mask"].to(self.device)

                tgt_text = x['target']['sentences']

                for decoder, name in zip(self.decoders, scorers):
                    kwargs = decoder.compute_decode_kwargs(input_ids, attention_mask)

                    start = time.perf_counter()
                    self._synchronize()
                    trans, iter = decoder.decode(input_ids, attention_mask, **kwargs)
                    self._synchronize()
                    end = time.perf_counter()
                    sample_time = end - start

                    init_tensor = kwargs["init_tensor"] if "init_tensor" in kwargs else ""

                    trans = self.dataset.tokenizer.batch_decode(
                        trans, skip_special_tokens=True
                    )[0]

                    scorers[name].update_metrics(sample_time, iter, trans, tgt_text[0], init_tensor)

                best_alg = self._compute_best_algorithm(scorers)
                best_algorithms[best_alg] += 1

                # Update tqdm bar
                postfix_pbar = self._compute_postfix(scorers, best_algorithms, best_alg)
                pbar.set_postfix(postfix_pbar)

                self.write_inline(i, scorers, best_alg)
            except Exception as e:
                if self.debug:
                    raise e
                else:
                    logger.error(e)
                return

            i += 1

        self.write_report(i, scorers, best_algorithms)
