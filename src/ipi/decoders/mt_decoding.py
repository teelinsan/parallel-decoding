from typing import Dict, Optional

import torch
from transformers import MBartForConditionalGeneration

from src.ipi import stopping_condition as sc
from src.utils.utils import get_logits_preprocessor

PREC_GOLD_AUTOREGRESSIVE: Dict[str, Optional[torch.Tensor]] = {"input_ids": None, "gold": None}
PREC_LOGISTS_PREPROCESSOR: Dict[str, Optional[torch.Tensor]] = {"input_ids": None, "logists": None}

class MTDecoder:
    def __init__(
            self,
            tokenizer,
            model,
            initializer,
            use_cache: bool = True,
            use_logits_preprocessor: bool = True,
            device: str = "cuda",
            **kwargs
    ):
        self.tokenizer = tokenizer

        self.initializer = initializer

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.use_cache = use_cache
        self.device = device

        with torch.no_grad():
            self.model = model
            self.model_name = self.model.name_or_path
            self.model.eval()

        self.max_length = min(self.tokenizer.model_max_length, 511)

        self.use_logits_preprocessor = use_logits_preprocessor

        self.is_mbart = isinstance(self.model, MBartForConditionalGeneration)

    def decode(self, input_ids, attention_mask, *args, **kwargs):
        pass

    def compute_decode_kwargs(self, *args, **kwargs):
        pass

    def initialize(self, **kwargs):
        pass

    def generate_gold_autoregressive(self, input_ids, attention_mask):

        global PREC_GOLD_AUTOREGRESSIVE

        if PREC_GOLD_AUTOREGRESSIVE['input_ids'] is None or not torch.equal(input_ids, PREC_GOLD_AUTOREGRESSIVE['input_ids']):
            if self.is_mbart:
                with self.tokenizer.as_target_tokenizer():
                    try:
                        lang_id = self.tokenizer.cur_lang_code_id
                    except:
                        lang_id = self.tokenizer.cur_lang_id
                gold_autoregressive = self.model.generate(
                    **{"input_ids": input_ids, "attention_mask": attention_mask},
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                    forced_bos_token_id=lang_id,
                )
            else:
                gold_autoregressive = self.model.generate(
                    **{"input_ids": input_ids, "attention_mask": attention_mask},
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                )
            gold_autoregressive = gold_autoregressive[:, : self.max_length]

            PREC_GOLD_AUTOREGRESSIVE['input_ids'] = input_ids
            PREC_GOLD_AUTOREGRESSIVE['gold'] = gold_autoregressive

        return PREC_GOLD_AUTOREGRESSIVE['gold']

    def generate_logits_preprocessor(self, input_ids):

        global PREC_LOGISTS_PREPROCESSOR

        if self.use_logits_preprocessor:
            if PREC_LOGISTS_PREPROCESSOR['input_ids'] is None or not torch.equal(input_ids, PREC_LOGISTS_PREPROCESSOR['input_ids']):
                logits_preprocessor = get_logits_preprocessor(
                    model=self.model,
                    input_ids=input_ids,
                    eos_token_id=self.eos_token_id
                )

                PREC_LOGISTS_PREPROCESSOR['input_ids'] = input_ids
                PREC_LOGISTS_PREPROCESSOR['logists'] = logits_preprocessor
        else:
            return None

        return PREC_LOGISTS_PREPROCESSOR['logists']

    @staticmethod
    def stopping_criterion(past_tensor, current_tensor, eos=None):
        return sc.stopping_criterion(past_tensor, current_tensor, eos)

    @staticmethod
    def limit_past_key_values(past_key_values, limit):
        return sc.limit_past_key_values(past_key_values, limit)

    @staticmethod
    def trig_eos(tensor, eos_token_id, init_tensor, base_value):
        if tensor[:, 0].item() == eos_token_id:
            return init_tensor[:, : base_value + 1]
        else:
            return None


def generate_target(
    tokenizer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    is_mbart: bool,
    decoding_method: str = "greedy",
    remove_padding: bool = False,
):
    if decoding_method == "greedy":
        if is_mbart:
            with tokenizer.as_target_tokenizer():
                gold_output = model.generate(
                    **{"input_ids": input_ids, "attention_mask": attention_mask},
                    num_beams=1,
                    do_sample=False,
                    forced_bos_token_id=tokenizer.cur_lang_code_id,
                )
        else:
            gold_output = model.generate(
                **{"input_ids": input_ids, "attention_mask": attention_mask},
                num_beams=1,
                do_sample=False,
            )
    else:
        raise NotImplementedError()

    if remove_padding:
        sample_lengths = (gold_output != tokenizer.pad_token_id).sum(dim=1)
        gold_output = [
            sample[:length] for sample, length in zip(gold_output, sample_lengths)
        ]

    return gold_output

