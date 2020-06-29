from torch import cat as pt_cat, zeros as pt_zeros, from_numpy, Tensor
def clotho_collate_fn_test(batch, nb_t_steps, input_pad_at):
    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower() == 'max' else min
        in_t_steps = truncate_fn([i[0].shape[0] for i in batch])
    else:
        in_t_steps = nb_t_steps

    in_dim = batch[0][0].shape[-1]

    input_tensor = []

    for in_b, filename in batch:
        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()
        input_tensor.append(tmp_in.unsqueeze_(0))

    input_tensor = pt_cat(input_tensor)

    filename = [i[1] for i in batch]

    return input_tensor, filename