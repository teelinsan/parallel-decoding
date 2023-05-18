import torch

from src.ipi.decoders.mt_decoding import MTDecoder
from src.viz.dependecy_graph import DecodingDependencyGraph


class JacobiDecoder(MTDecoder):
    def __init__(self, tokenizer, model, initializer, **kwargs):
        super().__init__(tokenizer, model, initializer, **kwargs)

        self.name = "jacobi"
        self.acronym = "j"

    def generate_target(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            is_mbart: bool,
            decoding_method: str = "greedy",
            remove_padding: bool = False,
    ):
        if decoding_method == "greedy":
            if is_mbart:
                with self.tokenizer.as_target_tokenizer():
                    gold_output = self.model.generate(
                        **{"input_ids": input_ids, "attention_mask": attention_mask},
                        num_beams=1,
                        do_sample=False,
                        forced_bos_token_id=self.tokenizer.cur_lang_code_id,
                    )
            else:
                gold_output = self.model.generate(
                    **{"input_ids": input_ids, "attention_mask": attention_mask},
                    num_beams=1,
                    do_sample=False,
                )
        else:
            raise NotImplementedError()

        if remove_padding:
            sample_lengths = (gold_output != self.tokenizer.pad_token_id).sum(dim=1)
            gold_output = [
                sample[:length] for sample, length in zip(gold_output, sample_lengths)
            ]

        return gold_output


    @torch.no_grad()
    def decode(
        self,
        input_ids,
        attention_mask,
        target_len=None,
        gold_target=None,
        init_tensor=None,
        compute_ddg: bool = False,
        logits_preprocessor=None,
        *args,
        **kwargs
    ):
        max_length = target_len
        str_index = 0
        key_cache = 0
        if compute_ddg:
            if gold_target is None:
                gold_target = self.generate_target(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    is_mbart=self.is_mbart,
                )
            ddg = DecodingDependencyGraph(
                model=self.model, tokenizer=self.tokenizer, gold_target=gold_target
            )

            max_length = gold_target.shape[-1]

        if init_tensor is None:
            init_tensor = torch.tensor(
                [self.pad_token_id] * input_ids.size(0) * max_length,
                device=self.device,
            ).reshape(input_ids.size(0), max_length)
        elif self.is_mbart:
            if init_tensor.shape[0] == 1:
                decoder_input_ids = init_tensor[:, 0].unsqueeze(0)
            else:
                decoder_input_ids = init_tensor[:, 0].unsqueeze(-1)
            output = self.model(
                input_ids,
                attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
            )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            past_key_values = output.past_key_values
            str_index = 1
            total_res = init_tensor
            init_tensor = init_tensor[:, 1:]
            key_cache = 1

        output_probs = init_tensor.clone().float()

        for index in range(str_index, max_length):
            if self.use_cache and index > 0:
                old_init_tensor = total_res.detach().clone()
                init_tensor = total_res[:, index:]
                output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids=init_tensor,
                    encoder_outputs=(encoder_last_hidden_state, None, None),
                    use_cache=True,
                    past_key_values=self.limit_past_key_values(past_key_values, index + key_cache),
                )
            else:
                old_init_tensor = init_tensor.detach().clone()
                output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids=init_tensor,
                    use_cache=True,
                )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            past_key_values = output.past_key_values
            logits = output.logits
            max_index = torch.argmax(logits, dim=-1)
            max_value, max_i = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            if index > 0 and logits_preprocessor is not None:
                logits_new = logits_preprocessor(total_res[:, : index + 1], logits[:, 0, :])
                max_value_new = torch.argmax(logits_new, dim=-1)
                max_index[:, 0] = max_value_new
            if self.use_cache and index > 0:
                init_tensor = max_index
                total_res = torch.cat(
                    (total_res[:, : index + 1], init_tensor[:, :-1]), dim=1
                )
            else:
                init_tensor[:, index + 1 :] = max_index[:, index:-1]
                total_res = init_tensor

                output_probs[:, index + 1 :] = max_value[:, index:-1]

            stop_condition, return_tensor = self.stopping_criterion(
                old_init_tensor, total_res
            )

            if compute_ddg:
                ddg.insert_one_element(
                    total_res, gold_target, output_probs=output_probs
                )

            if stop_condition:
                break

        if compute_ddg:
            return return_tensor, index, ddg

        return return_tensor, index


    def initialize(self, init_transl):
        if self.initializer is not None:
            init_tensor, _ = self.initializer.init_translation(init_transl.shape[-1])
        else:
            init_tensor = None

        return init_tensor

    def compute_decode_kwargs(self, input_ids, attention_mask, **kwargs):

        gold_autoregressive = self.generate_gold_autoregressive(input_ids, attention_mask)
        init_tensor = self.initialize(init_transl=gold_autoregressive)
        logits_preprocessor = self.generate_logits_preprocessor(input_ids)

        return {
            "init_tensor": init_tensor.clone(),
            "target_len": gold_autoregressive.shape[-1],
            "gold_target": gold_autoregressive,
            "logits_preprocessor": logits_preprocessor
        }
