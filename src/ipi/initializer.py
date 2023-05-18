import torch
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import (
    M2M100Tokenizer,
    MarianTokenizer,
    MBart50Tokenizer,
    MBartTokenizer,
)


class Initializer(object):
    def __init__(
        self,
        src_len,
        tgt_len,
        hugginface_tokenizer,
        use_init=True,
        device="cpu",
    ):

        self.src_len = src_len
        self.tgt_len = tgt_len
        self.tokenizer = hugginface_tokenizer

        self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer_nltk = ToktokTokenizer()
        self.detokenizer_nltk = TreebankWordDetokenizer()
        self.use_init = use_init
        self.device = device

    def init_translation(self, tgt_len=None):
        final_translation = ""
        with self.tokenizer.as_target_tokenizer():
            if isinstance(self.tokenizer, MBartTokenizer):
                tgt_tensor = self.tokenizer(
                    final_translation, return_tensors="pt", padding=True
                ).data["input_ids"]
                if tgt_tensor.shape[-1] == 2:
                    tgt_tensor = tgt_tensor[:, :1]
            elif isinstance(self.tokenizer, MarianTokenizer):
                bos = torch.tensor([self.pad_token_id]).unsqueeze(0)
                tgt_tensor = bos
            elif isinstance(self.tokenizer, MBart50Tokenizer) or isinstance(
                self.tokenizer, M2M100Tokenizer
            ):
                bos = torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)
                tgt_tensor = self.tokenizer(
                    final_translation, return_tensors="pt", padding=True
                ).data["input_ids"]
                tgt_tensor = torch.cat([bos, tgt_tensor], dim=-1)
            else:
                bos = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0)
                tgt_tensor = self.tokenizer(
                    final_translation, return_tensors="pt", padding=True
                ).data["input_ids"]
                tgt_tensor = torch.cat([bos, tgt_tensor], dim=-1)
        if tgt_len is not None:
            tgt_tensor = self.trim_length(tgt_tensor, tgt_len)
        return tgt_tensor.to(self.device), final_translation

    def trim_length(self, tgt_tensor, tgt_len):
        last_elem = tgt_tensor[:, -1].unsqueeze(0)
        if tgt_tensor.shape[-1] > tgt_len:
            return torch.cat([tgt_tensor[..., : tgt_len - 1], last_elem], dim=-1)
        elif tgt_tensor.shape[-1] < tgt_len:
            delta = tgt_len - tgt_tensor.shape[-1] - 1
            init_tensor = torch.tensor(
                [self.pad_token_id] * delta, dtype=tgt_tensor.dtype
            ).unsqueeze(0)
            return_tensor = torch.cat([tgt_tensor, init_tensor, last_elem], dim=-1)
            return return_tensor
        else:
            return tgt_tensor
