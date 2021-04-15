import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from torchviz import make_dot

try:
    import extend_profiler
except:
    pass

import repro
import pcl_bert

def compare(ref, opt, name=""):
    ref = ref.detach()
    opt = opt.detach()
    allclose = ref.allclose(opt, atol=1e-3, rtol=1e-4)
    print(f"{name}: ref: {ref.abs().mean():14g} allclose: {allclose}  shape: {ref.shape}")
    if not allclose:
      print(f"ref = {ref.view([-1])[:8]}, xsmm = {opt.view([-1])[:8]}")
      avg = ref.abs().mean()
      adiff = (ref - opt).abs()
      rdiff = adiff / avg
      err = 1e-6
      for ind, rd in np.ndenumerate(rdiff):
        if rd > err:
          print(f"{ind}: ref: {ref[ind].item():.7g} opt: {opt[ind].item():.7g} diff: {adiff[ind].item():.7g}  rdiff: {rd:.7g}")
          err = rd



if __name__ == '__main__':
    torch.manual_seed(1)
    B = 28
    S = 512
    N = 16
    H = 64
    p = 0.1
    iters=10
    bert_config = repro.BertConfig(is_decoder=False, attention_probs_dropout_prob=p, hidden_size=N*H, num_attention_heads=N)
    bsa_layer = repro.BertSelfAttention(bert_config)
    pbsa_layer = pcl_bert.BertSelfAttention(bert_config)

    #print(bsa_layer)
    #print(pbsa_layer)
    print([n for n,p in bsa_layer.named_parameters()])
    #print([n for n,p in pbsa_layer.named_parameters()])
    for ii, (i, o) in enumerate(zip(bsa_layer.parameters(), pbsa_layer.parameters())):
        o.data = i.data.clone().detach()
    bs = B
    seq_len = S
    hidden_size = bert_config.hidden_size
    inp = torch.empty([bs, seq_len, hidden_size]).uniform_(-1.0, 1.0)
    out = bsa_layer(inp)
    out1 = out[0].mean() * 100000
    out1.backward()
    pout = pbsa_layer(inp)
    out2 = pout[0].mean() * 100000
    out2.backward()
    inp1 = inp.clone().detach().requires_grad_()
    inp2 = inp.clone().detach().requires_grad_()
    einp1 = inp.clone().detach().requires_grad_()
    einp2 = inp.clone().detach().requires_grad_()
    print(f"hidden_states shape: {inp.shape}")
    print("Running Reference...")
    with torch.autograd.profiler.profile(True, False, record_shapes=True) as prof:
      for i in range(iters):
        out = bsa_layer(inp1, encoder_hidden_states=einp1)
        out1 = out[0].mean() * 100000
        out1.backward()
    file_prefix='reference'
    with open("%s.prof" % file_prefix, "w") as prof_f:
        prof_f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
    try:
        with open("%s.nested.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.nested_key_averages().table(sort_by="cpu_time_total"))
        with open("%s.top_level.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="cpu_time_total"))
        prof.print_op_timings(False, file_prefix)
    except:
        pass
    print("Running Optimized...")
    with torch.autograd.profiler.profile(True, False, record_shapes=True) as prof:
      for i in range(iters):
        pout = pbsa_layer(inp2, encoder_hidden_states=einp2)
        out2 = pout[0].mean() * 100000
        out2.backward()
    file_prefix='optimized'
    with open("%s.prof" % file_prefix, "w") as prof_f:
        prof_f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
    try:
        with open("%s.nested.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.nested_key_averages().table(sort_by="cpu_time_total"))
        with open("%s.top_level.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="cpu_time_total"))
        prof.print_op_timings(False, file_prefix)
    except:
        pass

    print("Done...")

