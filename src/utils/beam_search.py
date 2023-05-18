import csv
import os

import torch
import time
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, M2M100ForConditionalGeneration
import numpy as np

from src.utils.bench_scorer import Scorer
from src.utils.utils import makedirs, retrieve_model_name, write_sentences, get_logits_preprocessor


class BeamSearcher(object):
    def __init__(
            self,
            dataset,
            model,
            initializer,
            decoder,
            num_beams,
            no_repeat_ngram_size,
            early_stopping,
            batch_size,
            device,
            result_dir,
    ):
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        self.device = device
        self.result_dir = result_dir

        self.dataset = dataset
        self.initializer = initializer
        self.decoder = decoder

        self.tokenizer = dataset.tokenizer
        self.model = model
        model.eval().to(self.device)

        self.model_name = self.model.name_or_path
        print("model name in beam_search", self.model_name)

        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=False
        )

        if isinstance(
                self.model, MBartForConditionalGeneration
        ) or isinstance(
            self.model, M2M100ForConditionalGeneration
        ):
            self.is_mbart = True
        else:
            self.is_mbart = False

        self.exp_dir = self._retrieve_exp_dir()

    def _synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _retrieve_exp_dir(self):
        file_name = self._retrieve_file_name()

        exp_dir = os.path.join(self.result_dir, 'beam_search', file_name)
        makedirs(exp_dir)

        return exp_dir

    def _retrieve_file_name(self):
        model_name = retrieve_model_name(self.model_name.split("/")[1])
        lang = f"{self.dataset.src_lan}_{self.dataset.tgt_lan}"
        return f"{model_name}/{self.dataset.name}/{lang}"

    def _beam_search(self, input_ids, attention_mask):
      if self.is_mbart:
        with self.tokenizer.as_target_tokenizer():
            try:
                lang_id = self.tokenizer.cur_lang_code_id
            except:
                lang_id = self.tokenizer.cur_lang_id
        beam_output = self.model.generate(
            **{"input_ids": input_ids, "attention_mask": attention_mask},
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
            # no_repeat_ngram_size=self.no_repeat_ngram_size,
            forced_bos_token_id=lang_id,
            # do_sample=False,
            # use_cache=True,
        )
      else:
          beam_output = self.model.generate(
              **{"input_ids": input_ids, "attention_mask": attention_mask},
              num_beams=self.num_beams,
              early_stopping=self.early_stopping,
              # no_repeat_ngram_size=self.no_repeat_ngram_size,
              # do_sample=False,
              # use_cache=True,
          )

      return beam_output


    def _bench_time(self, input_ids, attention_mask):
        sample_time = []
        for _ in range(1):
          start = time.perf_counter()
          self._synchronize()
          beam_output = self._beam_search(input_ids, attention_mask)
          self._synchronize()
          end = time.perf_counter()
          sample_time.append(end - start)

        sample_mean = np.average(sample_time)
        sample_variance = np.var(sample_time)

        return sample_mean, sample_variance, beam_output

    def _auto_time(self, input_ids, attention_mask, logits_preprocessor=None):

        if self.initializer is not None:
            init_tensor, _ = self.initializer.init_translation()
        else:
            init_tensor = None

        sample_time = []
        for _ in range(1):
            init_new = init_tensor.clone()
            start = time.perf_counter()
            self._synchronize()
            auto_output, _ = self.decoder.autoregressive(
                input_ids, attention_mask, init_tensor=init_new, logits_preprocessor=logits_preprocessor
            )
            self._synchronize()
            end = time.perf_counter()
            sample_time.append(end - start)

        sample_mean = np.average(sample_time)
        sample_variance = np.var(sample_time)

        return sample_mean, sample_variance, auto_output

    def compute_beam_search(self, cfg):

        beam_scorer, auto_scorer = Scorer(), Scorer()
        worst_beam_translations = []

        pbar = tqdm(self.dataloader, desc="Computing Beam Search...")
        for x in pbar:
            input_ids = x["source"]["input_ids"].to(self.device)
            attention_mask = x["source"]["attention_mask"].to(self.device)

            tgt_text = x['target']['sentences']

            if cfg.model.use_logits_preprocessor:
                logits_preprocessor = get_logits_preprocessor(self.decoder, input_ids)
            else:
                logits_preprocessor = None

            mean_beam, var_beam, beam_output = self._bench_time(input_ids, attention_mask)
            mean_auto, var_auto, auto_output = self._auto_time(input_ids, attention_mask, logits_preprocessor)

            translation_beam = self.tokenizer.batch_decode(
                beam_output, skip_special_tokens=True
            )
            translation_auto = self.tokenizer.batch_decode(
                auto_output, skip_special_tokens=True
            )

            beam_scorer.update_metrics(mean_beam, var_beam, 0, translation_beam[0], tgt_text[0])
            auto_scorer.update_metrics(mean_auto, var_auto, 0, translation_auto[0], tgt_text[0])

            worst_beam_translations.extend(
                self._compute_tmp_bleu(
                    translation_beam[0],
                    translation_auto[0],
                    tgt_text[0],
                    beam_scorer.i
                )
            )

        self.write_report(beam_scorer, auto_scorer, worst_beam_translations)

    def write_report(self, beam_scorer, auto_scorer, worst_beam_translations):
        print("Writing report...")

        beam_score = beam_scorer.compute_bleu_score()
        auto_score = auto_scorer.compute_bleu_score()

        worst_beam_translations.sort(key=lambda x: x[1])

        # Table for the test info
        info_table = tabulate([
            ['Model', self.model_name],
            ['Dataset', self.dataset.name],
            ['Languages', f"{self.dataset.src_lan}-{self.dataset.tgt_lan}"],
            ['Device', self.device],
            ['Sentences', beam_scorer.i],
            ['Num Beams', self.num_beams],
            ['No Rep Ngram Size', self.no_repeat_ngram_size],
            ['Early Stopping', self.early_stopping],
            ['Do Sample', False],
            ['Use Cache', True],
        ], headers=['Info', 'Value'], tablefmt='grid')

        # Table for the Time
        time_table = tabulate([
            ['Time', beam_scorer.tot_mean_time, auto_scorer.tot_mean_time, (auto_scorer.tot_mean_time / beam_scorer.tot_mean_time)],
            ['Iter', beam_scorer.tot_mean_iter, auto_scorer.tot_mean_iter, 1],
            ['Var', beam_scorer.tot_var_time, auto_scorer.tot_var_time, 1],
        ], headers=['Metrics', 'Beam', 'Auto', 'Speedup'], tablefmt='grid')

        # Table for the Bleu score
        bleu_table = tabulate([
            ['Score', beam_score.score, auto_score.score],
            ['Counts', beam_score.counts, auto_score.counts],
            ['Totals', beam_score.totals, auto_score.totals],
            ['Precisions', beam_score.precisions, auto_score.precisions],
            ['Bp', beam_score.bp, auto_score.bp],
            ['Sys_len', beam_score.sys_len, auto_score.sys_len],
            ['ref_len', beam_score.ref_len, auto_score.ref_len],
        ], headers=['Metrics', 'Beam Search', 'Auto'], tablefmt='rst')

        with open(os.path.join(self.exp_dir, "report.txt"), mode='w') as report:
            report.write(f"Test Info\n{info_table}\n\n")
            report.write(f"Time\n{time_table}\n\n")
            report.write(f"Bleu Score\n{bleu_table}\n\n")

        with open(os.path.join(self.exp_dir, "worst_beam_translation.csv"), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['i', 'bleu', 'beam', 'auto', 'tgt'])
            for sample in worst_beam_translations:
                writer.writerow(list(sample))

        write_sentences(os.path.join(self.exp_dir, "beam.txt"), beam_scorer.predictions)
        write_sentences(os.path.join(self.exp_dir, "auto.txt"), auto_scorer.predictions)
        write_sentences(os.path.join(self.exp_dir, "reference.txt"), sum(auto_scorer.references, []))

    def _compute_tmp_bleu(self, translation_beam, translation_auto, tgt_text, i):

        beam_tmp_scorer, auto_tmp_scorer = Scorer(), Scorer()

        beam_tmp_scorer.update_metrics(0, 0, 0, translation_beam, tgt_text)
        auto_tmp_scorer.update_metrics(0, 0, 0, translation_auto, tgt_text)

        beam_score = beam_tmp_scorer.compute_bleu_score()
        auto_score = auto_tmp_scorer.compute_bleu_score()

        if beam_score.score < auto_score.score:
            return [(i, beam_score.score, translation_beam, translation_auto, tgt_text)]
        else:
            return []



