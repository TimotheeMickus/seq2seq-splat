import torch
import torch.nn as nn
import torch.nn.functional as F_
import transformers as hf

# indices for decomposition
I, S, T, F, C = range(5)

torch.autograd.set_grad_enabled(False)

def apply_layer_norm(ln_module, input_tensor, current_decomposition):
    # torch.allclose(ln(ipt), ln.bias + ((ln.weight * (ipt - ipt.mean(2, keepdims=True))) / (ipt.var(2, keepdims=True, unbiased=False) + ln.eps).sqrt()), atol=ln.eps) # should be True
    current_decomposition[...,C,:] += -input_tensor.mean(2, keepdims=True)
    current_decomposition = current_decomposition * ln_module.weight
    current_decomposition = current_decomposition / (input_tensor.var(2, keepdims=True, unbiased=False) + ln_module.eps).sqrt().unsqueeze(-1)
    current_decomposition[...,C,:] += ln_module.bias
    return (
        ln_module(input_tensor), # output
        current_decomposition,
    )

def apply_attention(attention_module, input_tensor, A_idx, attention_bank, attention_mask, current_decomposition):
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
    current_decomposition[:,:,A_idx,:] += attn_output
    current_decomposition[:,:,C,:] += attention_module.out_proj(attention_module.v_proj.bias)
    return (
        true_outputs,
        current_decomposition,
    )

def apply_ff(layer, input_tensor, current_decomposition):
    # Fully Connected
    hidden_states = layer.fc1(input_tensor)
    hidden_states = layer.activation_fn(hidden_states)
    raw_ff_term = F_.linear(hidden_states, layer.fc2.weight)
    true_hidden_states = raw_ff_term + layer.fc2.bias
    current_decomposition[:,:,F,:] += raw_ff_term
    current_decomposition[:,:,C,:] += layer.fc2.bias
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
        inputs_src = tokenizer([src_sent], return_tensors='pt', truncation=True).to(device)
        inputs_tgt = tokenizer([tgt_sent], return_tensors='pt', truncation=True).to(device)
        encoder_outputs = model.encoder(**inputs_src).last_hidden_state
        input_shape = inputs_tgt.input_ids.size()
        decoder = model.decoder
        decoder_embs = decoder.embed_tokens(inputs_tgt.input_ids) * decoder.embed_scale
        decoder_embs += decoder.embed_positions(input_shape, 0)
        input_tensor = decoder_embs
        decomposition = torch.zeros_like(decoder_embs).transpose(0,1).expand(-1, 5, -1).unsqueeze(0).contiguous()
        decomposition[:,:,I,:] += input_tensor

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
            hidden_state, current_decomposition = apply_attention(layer.self_attn, input_tensor, T, input_tensor, causal_mask, current_decomposition)
            hidden_state, current_decomposition = apply_layer_norm(layer.self_attn_layer_norm, hidden_state + input_tensor, current_decomposition)
            input_tensor = hidden_state

            hidden_state, current_decomposition = apply_attention(layer.encoder_attn, hidden_state, S, encoder_outputs, encoder_attn_mask, current_decomposition)
            hidden_state, current_decomposition = apply_layer_norm(layer.encoder_attn_layer_norm, hidden_state + input_tensor, current_decomposition)
            input_tensor = hidden_state

            hidden_state, current_decomposition = apply_ff(layer, hidden_state, current_decomposition)
            hidden_state, current_decomposition = apply_layer_norm(layer.final_layer_norm, hidden_state + input_tensor, current_decomposition)
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
