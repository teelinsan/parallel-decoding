import torch

from src.ipi.decoders.mt_decoding import MTDecoder


class AutoregressiveDecoder(MTDecoder):
    def __init__(self, tokenizer, model, initializer, **kwargs):
        super().__init__(tokenizer, model, initializer, **kwargs)

        self.name = "autoregressive"
        self.acronym = "a"

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, init_tensor=None, logits_preprocessor=None, *args, **kwargs):

        index = 0

        if init_tensor is None:
            init_tensor = torch.tensor(
                [self.pad_token_id], device=self.device
            ).unsqueeze(0)
        elif self.is_mbart:
            output = self.model(
                input_ids,
                attention_mask,
                decoder_input_ids=init_tensor[:, 0].unsqueeze(0),
                use_cache=True,
            )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            past_key_values = output.past_key_values
            index += 1
            total_res = torch.tensor(
                [[init_tensor[:, 0], init_tensor[:, 1]]], device=self.device
            )
            init_tensor = init_tensor[:, 1].unsqueeze(0)
        else:
            init_tensor = init_tensor[:, 0].unsqueeze(0)

        total_res = init_tensor.clone()
        while True:
            if self.use_cache and index > 0:
                if index == 1024:
                    print(total_res)
                output = self.model(
                    None,
                    attention_mask,
                    decoder_input_ids=init_tensor,
                    encoder_outputs=(encoder_last_hidden_state, None, None),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            else:
                output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids=init_tensor,
                    use_cache=True,
                )
            encoder_last_hidden_state = output.encoder_last_hidden_state
            past_key_values = output.past_key_values
            logits = output.logits
            if logits_preprocessor is not None:
                logits = logits_preprocessor(total_res, logits[:,-1,:])
            else:
                logits = logits[:,-1,:]
            max_value = torch.argmax(logits, dim=-1)
            last = max_value
            init_tensor = last.unsqueeze(0)
            total_res = torch.cat((total_res, init_tensor), dim=1)

            index += 1
            if last[0].item() == self.eos_token_id or index == self.model.config.max_length - 1:
                break
        return total_res, index

    def initialize(self):
        if self.initializer is not None:
            init_tensor, _ = self.initializer.init_translation()
        else:
            init_tensor = None

        return init_tensor

    def compute_decode_kwargs(self, input_ids, *args, **kwargs):
        init_tensor = self.initialize()
        logits_preprocessor = self.generate_logits_preprocessor(input_ids)

        return {
            "init_tensor": init_tensor.clone(),
            "logits_preprocessor": logits_preprocessor
        }

