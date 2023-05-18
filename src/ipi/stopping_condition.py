import torch


def limit_past_key_values(past_key_values, limit):
    new_list = []
    for elem in past_key_values:
        new_elem = list(elem)
        new_elem[0] = elem[0][:, :, :limit, :]
        new_elem[1] = elem[1][:, :, :limit, :]
        new_list.append(tuple(new_elem))
    return tuple(new_list)


def stopping_criterion(past_tensor, current_tensor, eos=None):
    assert past_tensor.shape == current_tensor.shape
    if torch.equal(past_tensor, current_tensor):
        tensor = current_tensor
        if eos is not None:
            if eos in current_tensor[0]:
                pos = (current_tensor[0] == eos).nonzero(as_tuple=True)[0]
                if pos.shape[0] > 1:
                    pos = pos[0].item()
                else:
                    pos = pos.item()
                return True, tensor, pos
            else:
                return True, tensor, -1
        return True, tensor
    else:
        if eos is not None:
            return False, current_tensor, False
        else:
            return False, current_tensor


def check_stop_cond(tensor, eos):
    if eos in tensor[0]:
        pos = (tensor[0] == eos).nonzero(as_tuple=True)[0]
        if pos.shape[0] > 1:
            pos = pos[0].item()
        else:
            pos = pos.item()
        return pos
    else:
        return -1
