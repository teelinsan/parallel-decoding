from src.ipi.decoders.mt_decoding import MTDecoder


class BeamSearchDecoder(MTDecoder):
    def __init__(self, tokenizer, model, initializer, num_beams, early_stopping, **kwargs):
        super().__init__(tokenizer, model, initializer, **kwargs)

        self.name = "beam_search"
        self.acronym = "b"

        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def decode(self, input_ids, attention_mask, *args, **kwargs):
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
                forced_bos_token_id=lang_id,
            )
        else:
            beam_output = self.model.generate(
                **{"input_ids": input_ids, "attention_mask": attention_mask},
                num_beams=self.num_beams,
                early_stopping=self.early_stopping,
            )

        return beam_output, 0

    def compute_decode_kwargs(self, *args, **kwargs):
        return {}