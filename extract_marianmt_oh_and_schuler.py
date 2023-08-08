import torch
import torch.nn as nn
import torch.nn.functional as F_
import transformers as hf

# indices for decomposition
S, T, C = range(3)

torch.autograd.set_grad_enabled(False)

def apply_layer_norm(ln_module, input_tensor, current_decomposition, centering_to_bias=False):
    # torch.allclose(ln(ipt), ln.bias + ((ln.weight * (ipt - ipt.mean(2, keepdims=True))) / (ipt.var(2, keepdims=True, unbiased=False) + ln.eps).sqrt()), atol=ln.eps) # should be True
    if centering_to_bias:
        current_decomposition[...,C,:] -= input_tensor.mean(2, keepdims=True)
    else:
        current_decomposition -= current_decomposition.mean(-1, keepdims=True)
    current_decomposition = current_decomposition * ln_module.weight
    current_decomposition = current_decomposition / (input_tensor.var(2, keepdims=True, unbiased=False) + ln_module.eps).sqrt().unsqueeze(-1)
    current_decomposition[...,C,:] += ln_module.bias
    return (
        ln_module(input_tensor), # output
        current_decomposition,
    )

def apply_source_attention(attention_module, input_tensor, attention_bank, attention_mask, current_decomposition):
    true_outputs, attention_weights, _ = attention_module(
        input_tensor,
        key_value_states=attention_bank,
        past_key_value=None,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    bsz, tgt_len, feats = input_tensor.size()
    src_len = attention_bank.size(1)
    proj_shape = (bsz * attention_module.num_heads, -1, attention_module.head_dim)
    value_states = F_.linear(attention_bank, attention_module.v_proj.weight)
    value_states = attention_module._shape(value_states, -1, bsz)
    attention_weights =  attention_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)
    attn_output = torch.bmm(attention_weights, value_states.view(*proj_shape))
    attn_output = attn_output.view(bsz, attention_module.num_heads, tgt_len, attention_module.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, attention_module.embed_dim)
    attn_output = F_.linear(attn_output, attention_module.out_proj.weight)
    current_decomposition[:,:,S,:] += attn_output
    current_decomposition[:,:,C,:] += attention_module.out_proj(attention_module.v_proj.bias)
    return (
        true_outputs,
        current_decomposition,
    )

def apply_self_attention(attention_module, input_tensor, attention_bank, attention_mask, current_decomposition):
    true_outputs, attention_weights, _ = attention_module(
        input_tensor,
        key_value_states=attention_bank,
        past_key_value=None,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    bsz, tgt_len, feats = input_tensor.size()
    src_len = attention_bank.size(1)
    proj_shape = (bsz * attention_module.num_heads, -1, attention_module.head_dim)
    dcp_proj_shape =  (bsz * attention_module.num_heads, 3, -1, attention_module.head_dim)
    dcp_value_states =  F_.linear(current_decomposition,  attention_module.v_proj.weight)

    dcp_value_states = dcp_value_states.view(bsz, -1, 3, attention_module.num_heads, attention_module.head_dim)
    dcp_value_states = dcp_value_states.transpose(1, 3).contiguous().view(*dcp_proj_shape)
    value_states = F_.linear(attention_bank, attention_module.v_proj.weight)
    value_states = attention_module._shape(value_states, -1, bsz)
    attention_weights =  attention_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)
    dcp_attention_weights =  attention_weights.unsqueeze(1)

    attn_output = torch.bmm(attention_weights, value_states.view(*proj_shape))
    dcp_attn_output = torch.einsum('bcxy,bcyf->bcxf', dcp_attention_weights.expand(-1, 3, -1, -1), dcp_value_states.view(*dcp_proj_shape))
    attn_output = attn_output.view(bsz, attention_module.num_heads, tgt_len, attention_module.head_dim)
    dcp_attn_output = dcp_attn_output.view(bsz, attention_module.num_heads, 3, tgt_len, attention_module.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, attention_module.embed_dim)
    dcp_attn_output = dcp_attn_output.transpose(1, 3).reshape(bsz, tgt_len, 3, attention_module.embed_dim)
    attn_output = F_.linear(attn_output, attention_module.out_proj.weight)
    dcp_attn_output = F_.linear(dcp_attn_output, attention_module.out_proj.weight)
    current_decomposition += dcp_attn_output
    current_decomposition[:,:,C,:] += attention_module.out_proj(attention_module.v_proj.bias)
    return (
        true_outputs,
        current_decomposition,
    )

def apply_ff(layer, input_tensor, current_decomposition):
    # Fully Connected
    with torch.autograd.set_grad_enabled(True):
        pre_act = layer.fc1(input_tensor.squeeze()).detach()
        pre_act.requires_grad = True
        post_act = layer.activation_fn(pre_act)
        post_act.sum().backward()  # populate gradient
        slopes = pre_act.grad.clone()
        intercepts = post_act - slopes * pre_act

    
    tmp =  current_decomposition[-1].clone()
    assert torch.allclose(tmp.sum(-2), input_tensor.squeeze())
    tmp = F_.linear(tmp, layer.fc1.weight)
    tmp[...,C,:] += layer.fc1.bias
    assert torch.allclose(tmp.sum(-2), pre_act)
    tmp = slopes.unsqueeze(1) * tmp
    tmp[...,C,:] += intercepts.squeeze()
    assert torch.allclose(tmp.sum(-2), post_act)
    tmp = F_.linear(tmp, layer.fc2.weight)
    tmp[...,C,:] += layer.fc2.bias
    current_decomposition[-1] += tmp

    raw_ff_term = F_.linear(post_act, layer.fc2.weight)
    true_hidden_states = raw_ff_term + layer.fc2.bias
    assert torch.allclose(tmp.sum(-2), true_hidden_states)
    
    return (
        true_hidden_states,
        current_decomposition,
    )


class Decomposer():
    def __init__(self, model_name, tokenizer_name, device='cuda'):
        self.model_name = model_name
        self.model = hf.MarianMTModel.from_pretrained(model_name).to(dtype=torch.float64, device=device)
        self.model.eval()
        self.tokenizer = hf.MarianTokenizer.from_pretrained(tokenizer_name)
        self.device = device

    def __call__(self, src_sent, tgt_sent, last_layer_only=True):
        model, tokenizer, device = self.model.model, self.tokenizer, self.device
        inputs_src = tokenizer([src_sent], return_tensors='pt').to(device)
        inputs_tgt = tokenizer([tgt_sent], return_tensors='pt').to(device)
        encoder_outputs = model.encoder(**inputs_src).last_hidden_state
        input_shape = inputs_tgt.input_ids.size()
        decoder = model.decoder
        decoder_embs = decoder.embed_tokens(inputs_tgt.input_ids) * decoder.embed_scale
        decoder_embs += decoder.embed_positions(input_shape, 0)
        input_tensor = decoder_embs
        decomposition = torch.zeros_like(decoder_embs).transpose(0,1).expand(-1, 3, -1).unsqueeze(0).contiguous()
        decomposition[:,:,T,:] += input_tensor

        if not last_layer_only:
            all_decompositions = [decomposition]
            current_decomposition = decomposition.detach().clone()
        else:
            current_decomposition = decomposition
        assert torch.allclose(current_decomposition.sum(-2), input_tensor), 'Imprecise decomposition!'
        causal_mask = decoder._prepare_decoder_attention_mask(
            None, input_shape, decoder_embs, 0
        )
        encoder_attn_mask = hf.models.marian.modeling_marian._expand_mask(inputs_src.attention_mask, decomposition.dtype, tgt_len=input_shape[-1])
        # hf.models.marian.modeling_marian._make_causal_mask(inputs_tgt.input_ids.size(), input_tensor.dtype)
        for idx, layer in enumerate(decoder.layers):
            reference = layer(
                input_tensor,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_attn_mask,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=True,
                use_cache=False,
            )[0]
            
            residual = current_decomposition.detach().clone()
            hidden_state, current_decomposition = apply_self_attention(layer.self_attn, input_tensor, input_tensor, causal_mask, current_decomposition)
            hidden_state, current_decomposition = apply_layer_norm(layer.self_attn_layer_norm, hidden_state + input_tensor, current_decomposition)
            input_tensor = hidden_state
            assert torch.allclose(current_decomposition.sum(-2), input_tensor)

            hidden_state, current_decomposition = apply_source_attention(layer.encoder_attn, hidden_state, encoder_outputs, encoder_attn_mask, current_decomposition)
            hidden_state, current_decomposition = apply_layer_norm(layer.encoder_attn_layer_norm, hidden_state + input_tensor, current_decomposition)
            input_tensor = hidden_state
            assert torch.allclose(current_decomposition.sum(-2), input_tensor)

            hidden_state, current_decomposition = apply_ff(layer, hidden_state, current_decomposition)
            assert torch.allclose(current_decomposition.sum(-2), hidden_state + input_tensor)
            hidden_state, current_decomposition = apply_layer_norm(layer.final_layer_norm, hidden_state + input_tensor, current_decomposition)
            assert torch.allclose(current_decomposition.sum(-2), hidden_state)
            input_tensor = hidden_state

            if not last_layer_only:
                all_decompositions.append(current_decomposition)
                current_decomposition = current_decomposition.detach().clone()
            assert torch.allclose(current_decomposition.sum(-2), reference), 'Imprecise decomposition!'
            assert torch.allclose(input_tensor, reference), 'Imprecise decomposition!'

        assert torch.allclose(decoder(**inputs_tgt, encoder_hidden_states=encoder_outputs).last_hidden_state, current_decomposition.sum(-2)), 'Imprecise decomposition!'
        if not last_layer_only:
            current_decomposition = torch.cat(all_decompositions, dim=0)
        return current_decomposition
