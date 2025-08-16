import torch
import pdb

def top_p(data: torch.Tensor, p):
    """
    algorithm
        use torch.topk to find a range rougthly, and then cumulate the probability, use binary-search to 
        find the p boundary
    """
    max_k = data.shape[-1]
    k = min(max_k, 2000)
    top_k, indices = torch.topk(data, k)    # top_k has been sorted by descending order
    cumsum = torch.cumsum(top_k, dim=-1)
    if not torch.all(cumsum[:, -1] > p).item():     # if k is not large enough to cover cumulated probability p 
        top_k, indices = torch.topk(data, max_k)
        cumsum = torch.cumsum(top_k, dim=-1)
    
    # pdb.set_trace()
    # sorted_cum_topk, _ = torch.sort(cumsum, dim=-1, descending=True) 
    bs, max_len = cumsum.shape
    targets = torch.searchsorted(cumsum, torch.ones((bs, 1))*p)
    print(targets)
    positions = torch.arange(max_len).unsqueeze(0).expand(bs, -1)
    top_k = top_k.masked_fill(positions > targets, 0)
    print(top_k)
    samples_idx = torch.multinomial(top_k, 1)
    res_indices = torch.gather(indices, -1, samples_idx)
    res_values = torch.gather(data, -1, res_indices)
    return res_values, res_indices


t = torch.softmax(torch.randn((2, 10)), dim=-1)
v, i = top_p(t, 0.7)
print(t)
print(v)
print(i)

