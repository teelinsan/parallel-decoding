import torch
from more_itertools import sliced

from src.ipi.decoders.mt_decoding import MTDecoder


class HybridJacobiDecoder(MTDecoder):
    def __init__(self, tokenizer, model, gs_jaco_blocks, init_mode, initializer, percent=300, **kwargs):
        super().__init__(tokenizer, model, initializer, **kwargs)

        self.name = "hybrid_jacobi"
        self.acronym = "h"

        self.gs_jaco_blocks = gs_jaco_blocks

        self.init_mode = init_mode
        self.percent = percent

    @torch.no_grad()
    def decode(
        self, input_ids, attention_mask, target_len, gold_target, init_tensor=None, logits_preprocessor=None, *args, **kwargs
    ):
        key_cache = 1
        if init_tensor is None:
            init_tensor = torch.tensor(
                [self.pad_token_id] * target_len, device=self.device
            )
            blocks = list(sliced(init_tensor, self.gs_jaco_blocks))
            init_tensor = init_tensor.unsqueeze(0)
            total_past_key_values = None
        elif self.is_mbart:
            output = self.model(
                input_ids,
                attention_mask,
                decoder_input_ids=init_tensor[:, 0].unsqueeze(0),
                use_cache=True,
            )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            total_past_key_values = output.past_key_values
            init_tensor = init_tensor[:, 1:]
            blocks = list(sliced(init_tensor.squeeze(0), self.gs_jaco_blocks))
            key_cache = 2
        else:
            init_tensor = init_tensor
            blocks = list(sliced(init_tensor.squeeze(0), self.gs_jaco_blocks))
            total_past_key_values = None

        iteration_saved = 0
        base_value = 0

        for blocco in blocks:
            max_len = blocco.shape[-1]
            blocco_usr = init_tensor[:, base_value : base_value + max_len]
            for index in range(max_len):
                old_blocco = blocco_usr.detach().clone()
                trig = self.trig_eos(
                    old_blocco, self.eos_token_id, init_tensor, base_value
                )
                if trig is not None:
                    return trig, (gold_target.shape[-1] - 1) - iteration_saved
                blocco_usr_new = blocco_usr[:, index:]
                if base_value == 0 and index == 0 and not self.is_mbart:
                    output = self.model(
                        input_ids,
                        attention_mask,
                        decoder_input_ids=blocco_usr_new,
                        use_cache=True,
                        past_key_values=total_past_key_values,
                    )
                    encoder_last_hidden_state = output.encoder_last_hidden_state
                else:
                    output = self.model(
                        input_ids,
                        attention_mask,
                        decoder_input_ids=blocco_usr_new,
                        encoder_outputs=(encoder_last_hidden_state, None, None),
                        use_cache=True,
                        past_key_values=total_past_key_values,
                    )

                total_past_key_values = self.limit_past_key_values(
                    output.past_key_values,
                    base_value + index + key_cache,
                )

                logits = output.logits
                max_value = torch.argmax(logits, dim=-1)

                if logits_preprocessor is not None:
                    logits_new = logits_preprocessor(init_tensor[:, :base_value + index + 1], logits[:, 0, :])
                    max_value_new = torch.argmax(logits_new, dim=-1)
                    max_value[:,0] = max_value_new

                if (
                    max_value.shape[-1]
                    == init_tensor[
                        :, base_value + index + 1 : base_value + max_len + 1
                    ].shape[-1]
                ):
                    init_tensor[
                        :, base_value + index + 1 : base_value + max_len + 1
                    ] = max_value[:, :]
                else:
                    # If last block remove the last token after EOS
                    init_tensor[
                        :, base_value + index + 1 : base_value + max_len + 1
                    ] = max_value[:, :-1]

                stop_condition, _, eos_cond = self.stopping_criterion(
                    old_blocco, blocco_usr, eos=self.eos_token_id
                )

                if stop_condition:
                    if eos_cond >= 0:
                        return (
                            init_tensor[:, : base_value + eos_cond + 1],
                            (gold_target.shape[-1] - 1) - iteration_saved,
                        )
                    if index + 1 != max_len:
                        iteration_saved += max_len - index - 1
                        total_past_key_values = self.limit_past_key_values(
                            output.past_key_values,
                            base_value + max_len + 1,
                        )
                        break

            base_value += max_len

        total_res, total_iter = (
            init_tensor,
            (gold_target.shape[-1] - 1) - iteration_saved,
        )

        init_tensor = init_tensor[:, -1].clone().unsqueeze(0)

        #Go autoregressive until [EOS]
        while True and base_value != self.model.config.max_length - 1:
            index = 0
            output = self.model(
                input_ids,
                attention_mask,
                decoder_input_ids=init_tensor,
                encoder_outputs=(encoder_last_hidden_state, None, None),
                use_cache=True,
                past_key_values=total_past_key_values,
            )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            total_past_key_values = output.past_key_values
            logits = output.logits
            max_value = torch.argmax(logits, dim=-1)
            last = max_value[:, -1]
            if self.use_cache:
                init_tensor = last.unsqueeze(0)
                total_res = torch.cat((total_res, init_tensor), dim=1)

            index += 1
            if last[0].item() == self.eos_token_id:
                break
        return total_res, index + total_iter

    def initialize(self, input_ids, gold_autoregressive):
        if self.initializer is not None:
            if self.init_mode == "under":
                len = max(3, input_ids.shape[-1] - self.percent / 100 * input_ids.shape[-1])
                m = int(len)
            elif self.init_mode == "over":
                len = input_ids.shape[-1] + self.percent / 100 * input_ids.shape[-1]
                m = int(len)
            elif self.init_mode == "fixed":
                m = 511
            else:
                m = gold_autoregressive.shape[-1]
            init_tensor, _ = self.initializer.init_translation(m)
        else:
            init_tensor = None

        return init_tensor

    def compute_decode_kwargs(self, input_ids, attention_mask, **kwargs):
        gold_autoregressive = self.generate_gold_autoregressive(input_ids, attention_mask)
        init_tensor = self.initialize(input_ids, gold_autoregressive)
        logits_preprocessor = self.generate_logits_preprocessor(input_ids)

        return{
            "init_tensor": init_tensor.clone(),
            "gold_target": gold_autoregressive,
            "target_len": gold_autoregressive.shape[-1],
            "logits_preprocessor": logits_preprocessor
        }