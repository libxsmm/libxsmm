import math
from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from blocked_layout import BlockedParameter, BlockedModule, BlockedTensor
import blocked_layout
import pcl_bert_ext
import time
from contextlib import contextmanager

pcl_bert_ext.init_libxsmm()
pcl_bert_ext.reset_debug_timers()

USE_BF16_PARAMS = True

def reset_debug_timers():
    pcl_bert_ext.reset_debug_timers()

def print_debug_timers(tid=0):
    pcl_bert_ext.print_debug_timers(tid)

def set_rnd_seed(seed):
    pcl_bert_ext.set_rnd_seed(seed)

class DummyLinear(BlockedModule):
    def __init__(self, in_features, out_features, bias=True):
        super(DummyLinear, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        raise NotImplemented
        return input

class BertSelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, training, need_attention_output, *inputs):
        #print("FWD Called")
        #print("BSAFWD:", [t.shape if isinstance(t, torch.Tensor) else t for t in inputs[6:]])
        context_layer, attention_probs_out, hs_t, ehs_t, ql_t, kl_tv, vl_tv, ap, apd_t, ap_dp_mask = pcl_bert_ext.fused_self_attention_fwd(p, inputs, training)
        (qw, qb, kw, kb, vw, vb, hs, am, hm, ehs, eam) = inputs
        ctx.save_for_backward(qw, kw, vw, hs_t, hm, ehs_t, ql_t, kl_tv, vl_tv, ap, apd_t, ap_dp_mask)
        ctx.p = p
        # stop = False
        # for i, t in enumerate([context_layer, attention_probs_out, hs_t, ehs_t, ql_t, kl_tv, vl_tv, ap, apd_t, ap_dp_mask]):
        #    nan = t.isnan().any().item()
        #    stop = stop or nan
        #    if nan: print ("Nan found in %d tensor" % i)
        # if stop: raise "Nan Found"

        #print("Returning from FWD")
        if need_attention_output:
            return context_layer, attention_probs_out
        else:
            return (context_layer,)

    @staticmethod
    def backward(ctx, *grad_outs):
        #print("BWD Called")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]
        if len(inputs) == 1: inputs.append(inputs[0].new_empty(0))
        inputs += ctx.saved_tensors
        p = ctx.p
        (dqw, dqb, dkw, dkb, dvw, dvb, dhs, dehs) = pcl_bert_ext.fused_self_attention_bwd(p, inputs)
        ehs = inputs[7]
        if ehs is None: dehs = None
        #print("Returning from BWD")
        #print("DHS:", dhs.view([-1])[:4])
        return (None, None, None, dqw, dqb, dkw, dkb, dvw, dvb, dhs, None, None, dehs, None)

class BertSelfAttention(BlockedModule):
    r"""PCL Bert Self Attention Layer using libxsmm blocked GEMM"""

    #__constants__ = ['bias', 'C', 'K']

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        #self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads # N
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # H
        self.all_head_size = self.num_attention_heads * self.attention_head_size #NH
        self.hidden_size = config.hidden_size #HS
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.query = DummyLinear(config.hidden_size, self.all_head_size)
        self.key = DummyLinear(config.hidden_size, self.all_head_size)
        self.value = DummyLinear(config.hidden_size, self.all_head_size)
        self.query.weight.set_blocking_param(([self.attention_head_size, self.attention_head_size], [0, 2, 3, 1],))
        self.key.weight.set_blocking_param(([self.attention_head_size, self.attention_head_size], [0, 2, 3, 1],))
        self.value.weight.set_blocking_param(([self.attention_head_size, self.attention_head_size], [0, 2, 3, 1],))
        self.blocked_input_signature = blocked_layout.get_blocking_signature("BSF", "BSFSF")
        self.use_bf16 = False

        #self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def maybe_block_params(self):
        self.query.weight.block()
        self.key.weight.block()
        self.value.weight.block()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self.maybe_block_params()
        if encoder_hidden_states is not None:
            assert encoder_hidden_states.shape == hidden_states.shape, "Different shapes not supported(%s != %s)" % (encoder_hidden_states.shape, hidden_states.shape)
            encoder_hidden_states = self.get_blocked_tensor(encoder_hidden_states, self.blocked_input_signature, [None, None, self.attention_head_size])
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.attention_head_size])
        #print(f"hidden_states: {hidden_states.shape}")
        inputs = [self.query.weight, self.query.bias,
            self.key.weight, self.key.bias,
            self.value.weight, self.value.bias]
        inputs.append(hidden_states)
        if attention_mask is not None:
            #print(f"attention_mask: {attention_mask.shape}")
            #B, S1, N, S2, H = hidden_states.shape
            #S = S1 * S2
            #print("Before attention_mask shape = %s (%s)" % (attention_mask.shape, attention_mask.numel()))
            #attention_mask = attention_mask.expand([B, N, S, S]).view([B, N, S1, S2, S1, S2]).permute([0, 2, 1, 4, 3, 5]).contiguous()
            assert attention_mask.size(1) == attention_mask.size(2) == 1, "unsupported attention_mask shape %s" % (attention_mask.shape,)
            attention_mask = attention_mask.contiguous()
            #print("After  attention_mask shape = %s (%s)" % (attention_mask.shape, attention_mask.numel()))
        if head_mask is not None:
            print(f"head_mask: {head_mask.shape}")
        if encoder_attention_mask is not None:
            print(f"encoder_attention_mask: {encoder_attention_mask.shape}")
            #B, S1, N, S2, H = encoder_hidden_states.shape
            #S = S1 * S2
            #encoder_attention_mask = encoder_attention_mask.expand([B, N, S, S]).view([B, N, S1, S2, S1, S2]).permute([0, 2, 1, 4, 3, 5]).contiguous()
            assert encoder_attention_mask.size(1) == encoder_attention_mask.size(2) == 1, "unsupported encoder_attention_mask shape %s" % (encoder_attention_mask.shape,)
            encoder_attention_mask = encoder_attention_mask.contiguous()
        inputs.append(attention_mask if attention_mask is not None else torch.Tensor())
        inputs.append(head_mask if head_mask is not None else torch.Tensor() )
        inputs.append(encoder_hidden_states if encoder_hidden_states is not None else torch.Tensor())
        inputs.append(encoder_attention_mask if encoder_attention_mask is not None else torch.Tensor())

        # context_layer, attention_probs = pcl_bert_ext.forward(self.handle.handle, inputs)
        p = self.attention_probs_dropout_prob if self.training else 0.0
        if self.use_bf16:
            inputs = [i.to(torch.bfloat16) if i.is_floating_point() else i for i in inputs ]
        outputs = BertSelfAttentionFunction.apply(p, self.training, output_attentions, *inputs)
        #outputs = BertSelfAttentionFunction.apply(p, self.training, True, *inputs)
        context_layer = outputs[0]

        context_layer = BlockedTensor(context_layer, self.blocked_input_signature, orig_hidden_states.dtype)
        if output_attentions:
            print("Reshaping output_attentions")
            attention_probs = outputs[1]
            attention_probs = attention_probs.permute([0, 2, 1, 4, 3, 5]).contiguous().view([B, self.num_attention_heads, S, S]).to(orig_hidden_states.dtype)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class BertSelfAttentionBF16(BertSelfAttention):
    r"""PCL Bert Self Attention BF16 Layer using libxsmm blocked GEMM"""

    def __init__(self, config):
        super().__init__(config)
        if USE_BF16_PARAMS:
            self.query.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
            self.key.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
            self.value.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
        self.use_bf16 = True

class BertOutputBaseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, eps, training, *inputs):
        (inp, inp2, wt, bias, gamma, beta) = inputs
        #print("A")
        outputs = pcl_bert_ext.fused_dense_dropout_layernorm_fwd(p, eps, inputs, training)
        #print("B")
        (out, dout, mean, var, dp_mask) = outputs
        ctx.save_for_backward(inp, wt, gamma, mean, var, dout, dp_mask)
        #print("C")
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp, grad_inp2, grad_wt, grad_bias, grad_gamma, grad_beta = pcl_bert_ext.fused_dense_dropout_layernorm_bwd(ctx.p, inputs)
        return (None, None, None, grad_inp, grad_inp2, grad_wt, grad_bias, grad_gamma, grad_beta)

class BertOutputBase(BlockedModule):
    def __init__(self, config, selfOutput):
        super().__init__()
        ifm = config.hidden_size if selfOutput else config.intermediate_size
        self.dense = DummyLinear(ifm, config.hidden_size)
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense.weight.set_blocking_param(([self.attention_head_size, self.attention_head_size], [0, 2, 3, 1],))
        self.blocked_input_signature = blocked_layout.get_blocking_signature("BSF", "BSFSF")
        self.use_bf16 = False
        print(f"config.hidden_size = {config.hidden_size}, ifm = {ifm}, p = {config.hidden_dropout_prob}, eps = {config.layer_norm_eps}")

    def maybe_block_params(self):
        self.dense.weight.block()

    def forward(self, hidden_states, input_tensor):
        self.maybe_block_params()
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.attention_head_size])
        input_tensor = self.get_blocked_tensor(input_tensor, self.blocked_input_signature, [None, None, self.attention_head_size])

        inputs = [hidden_states, input_tensor, self.dense.weight, self.dense.bias,
                    self.LayerNorm.weight, self.LayerNorm.bias]
        p = self.hidden_dropout_prob if self.training else 0.0
        if self.use_bf16:
            inputs = [i.to(torch.bfloat16) if i.is_floating_point() else i for i in inputs ]
        ret = BertOutputBaseFunction.apply(p, self.layer_norm_eps, self.training, *inputs)
        #ret = ret.to(hidden_states.dtype)
        ret = BlockedTensor(ret, self.blocked_input_signature, orig_hidden_states.dtype)
        return ret
        #hidden_states = self.dense(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #return hidden_states

class BertSelfOutput(BertOutputBase):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__(config, True)

class BertSelfOutputBF16(BertOutputBase):
    def __init__(self, config):
        super(BertSelfOutputBF16, self).__init__(config, True)
        if USE_BF16_PARAMS:
            self.dense.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
        self.use_bf16 = True

class BertOutput(BertOutputBase):
    def __init__(self, config):
        super(BertOutput, self).__init__(config, False)

class BertOutputBF16(BertOutputBase):
    def __init__(self, config):
        super(BertOutputBF16, self).__init__(config, False)
        if USE_BF16_PARAMS:
            self.dense.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
        self.use_bf16 = True

class BertIntermediateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, act, training):
        #assert act == "gelu_new", "%s activation type is not supported" % act
        gelu_in, output = pcl_bert_ext.fused_dense_gelu_fwd(input, weight, bias, training)
        ctx.save_for_backward(input, weight, gelu_in)
        ctx.act = act
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (input, weight, gelu_in) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_inp, grad_wt, grad_bias = pcl_bert_ext.fused_dense_gelu_bwd(grad_out, gelu_in, input, weight)
        return (grad_inp, grad_wt, grad_bias, None, None)

class BertIntermediate(BlockedModule):
    def __init__(self, config):
        super().__init__()
        self.dense = DummyLinear(config.hidden_size, config.intermediate_size)
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dense.weight.set_blocking_param(([self.attention_head_size, self.attention_head_size], [0, 2, 3, 1],))
        assert config.hidden_act in ["gelu", "gelu_new"], "Currently, only GELU new is supported in fused op, %s is given" % config.hidden_act
        self.hidden_act = config.hidden_act
        self.blocked_input_signature = blocked_layout.get_blocking_signature("BSF", "BSFSF")
        self.use_bf16 = False
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def maybe_block_params(self):
        self.dense.weight.block()

    def forward(self, hidden_states):
        self.maybe_block_params()
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.attention_head_size])
        inputs = [hidden_states, self.dense.weight, self.dense.bias]
        if self.use_bf16:
            inputs = [i.to(torch.bfloat16) if i.is_floating_point() else i for i in inputs ]
        ret = BertIntermediateFunction.apply(*inputs, self.hidden_act, self.training)
        #ret = ret.to(hidden_states.dtype)
        hidden_states = BlockedTensor(ret, self.blocked_input_signature, orig_hidden_states.dtype)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertIntermediateBF16(BertIntermediate):
    def __init__(self, config):
        super(BertIntermediateBF16, self).__init__(config)
        if USE_BF16_PARAMS:
            self.dense.weight.set_blocking_param(([self.attention_head_size, [self.attention_head_size//2, 2]], [0, 2, 3, 1, 4], torch.bfloat16,))
        self.use_bf16 = True

class BertEmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, prob, eps, head_size, pad_id, *inputs):
        (ii, pi, ti, ie, g, b, we, pe, te) = inputs
        (out, eout, mean, var, msk) = pcl_bert_ext.fused_embedding_layernorm_dropout_fwd(prob, eps, head_size, pad_id, inputs, training)
        ctx.save_for_backward(ii, pi, ti, ie, g, we, pe, te, mean, var, eout, msk)
        ctx.prob = prob
        ctx.pad_id = pad_id
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
      prob = ctx.prob
      pad_id = ctx.pad_id
      inputs = []
      inputs += grad_outs
      inputs += ctx.saved_tensors
      (die, dg, db, dwe, dpe, dte) = pcl_bert_ext.fused_embedding_layernorm_dropout_bwd(prob, pad_id, inputs)
      grad_inps = (None, None, None, die, dg, db, dwe, dpe, dte,)
      return (None, None, None, None, None) + grad_inps

class BertEmbeddings(BlockedModule):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_eps = config.layer_norm_eps
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.pad_token_id = config.pad_token_id

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.blocked_ids_signature = blocked_layout.get_blocking_signature("BS", "BSS")
        self.blocked_embed_signature = blocked_layout.get_blocking_signature("BSF", "BSFSF")
        self.use_bf16 = False

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = self.get_blocked_tensor(input_ids, self.blocked_ids_signature, [None, None])
        else:
            input_shape = inputs_embeds.size()[:-1]
            input_ids = torch.LongTensor()
            inputs_embeds = self.get_blocked_tensor(inputs_embeds, self.blocked_embed_signature, [None, None, self.attention_head_size])

        #seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.LongTensor()
        else:
            position_ids = self.get_blocked_tensor(position_ids, self.blocked_ids_signature, [None, None])

        if token_type_ids is None:
            token_type_ids = torch.LongTensor()
        else:
            token_type_ids  = self.get_blocked_tensor(token_type_ids , self.blocked_ids_signature, [None, None])

        if inputs_embeds is None:
            inputs_embeds = torch.Tensor()
        #     inputs_embeds = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        emb_weighs = [self.word_embeddings.weight, self.position_embeddings.weight, self.token_type_embeddings.weight]
        inputs = [input_ids, position_ids, token_type_ids, inputs_embeds, self.LayerNorm.weight, self.LayerNorm.bias]
        p = self.hidden_dropout_prob if self.training else 0.0
        if self.use_bf16:
            inputs = [i.to(torch.bfloat16) if i.is_floating_point() else i for i in inputs ]
        inputs += emb_weighs
        embeddings = BertEmbeddingsFunction.apply(self.training, p, self.layer_norm_eps, self.attention_head_size, self.pad_token_id, *inputs)
        #embeddings = BlockedTensor(embeddings, self.blocked_embed_signature, torch.bfloat16 if self.use_bf16 else torch.float)
        embeddings = BlockedTensor(embeddings, self.blocked_embed_signature, torch.float)
        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddingsBF16(BertEmbeddings):
    def __init__(self, config):
        super(BertEmbeddingsBF16, self).__init__(config)
        self.use_bf16 = True

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Lower bits for bf16 params
                    if p.data.dtype == torch.bfloat16:
                        state["low_bits"] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if p.data.dtype == torch.bfloat16: low_bits = state["low_bits"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # # Decay the first and second moment running average coefficient
                # # In-place operations to update the averages at the same time
                # exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # # Just adding the square of the weights to the loss function is *not*
                # # the correct way of using L2 regularization/weight decay with Adam,
                # # since that will interact with the m and v parameters in strange ways.
                # #
                # # Instead we want to decay the weights in a manner that doesn't interact
                # # with the m/v parameters. This is equivalent to adding the square
                # # of the weights to the loss with plain (non-momentum) SGD.
                # # Add weight decay at the end (fixed version)
                # if group["weight_decay"] > 0.0:
                #     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                if p.data.dtype == torch.bfloat16:
                    pcl_bert_ext.fused_split_adamw(p.data, low_bits, grad.contiguous(), exp_avg, exp_avg_sq, beta1, beta2, step_size, group["lr"], group["weight_decay"], group["eps"])
                else:
                    pcl_bert_ext.fused_adamw(p.data, grad.contiguous(), exp_avg, exp_avg_sq, beta1, beta2, step_size, group["lr"], group["weight_decay"], group["eps"])

        return loss



def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    from torch._six import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == 2:
        return pcl_bert_ext.clip_grad_norm([p.grad.detach() for p in parameters], max_norm)

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

# bm_default_blocking_factors = BlockedModule.default_blocking_factors
# @staticmethod
# def custom_blocking_factors(S):
#     print(f"S = {S}")
#     if S % 32 == 0: return [S//32, 32]
#     return bm_default_blocking_factors
# BlockedModule.default_blocking_factors = custom_blocking_factors

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    try:
        import transformers
        orig_BertSelfAttention = transformers.models.bert.modeling_bert.BertSelfAttention
        orig_BertSelfOutput = transformers.models.bert.modeling_bert.BertSelfOutput
        orig_BertOutput = transformers.models.bert.modeling_bert.BertOutput
        orig_BertIntermediate = transformers.models.bert.modeling_bert.BertIntermediate
        orig_BertEmbeddings = transformers.models.bert.modeling_bert.BertEmbeddings
        try:
            if enable:
                if use_bf16:
                    transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttentionBF16
                    transformers.models.bert.modeling_bert.BertSelfOutput = BertSelfOutputBF16
                    transformers.models.bert.modeling_bert.BertOutput = BertOutputBF16
                    transformers.models.bert.modeling_bert.BertIntermediate = BertIntermediateBF16
                    transformers.models.bert.modeling_bert.BertEmbeddings = BertEmbeddingsBF16
                else:
                    transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention
                    transformers.models.bert.modeling_bert.BertSelfOutput = BertSelfOutput
                    transformers.models.bert.modeling_bert.BertOutput = BertOutput
                    transformers.models.bert.modeling_bert.BertIntermediate = BertIntermediate
                    transformers.models.bert.modeling_bert.BertEmbeddings = BertEmbeddings
            yield
        finally:
            transformers.models.bert.modeling_bert.BertSelfAttention = orig_BertSelfAttention
            transformers.models.bert.modeling_bert.BertSelfOutput = orig_BertSelfOutput
            transformers.models.bert.modeling_bert.BertOutput = orig_BertOutput
            transformers.models.bert.modeling_bert.BertIntermediate = orig_BertIntermediate
            transformers.models.bert.modeling_bert.BertEmbeddings = orig_BertEmbeddings
    except ImportError as e:
        pass
