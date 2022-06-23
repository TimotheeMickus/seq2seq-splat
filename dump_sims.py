import csv
import pathlib

import torch
import tqdm
import transformers as hf

from extract_marianmt import Decomposer

def spim(decomposition, total_embs=None):
    if total_embs is None:
        total_embs = decomposition.sum(-2)
    norms = torch.linalg.norm(total_embs, dim=-1, keepdims=True) ** 2
    scalar_prods = torch.einsum('lstd,lsd->lst', decomposition, total_embs)
    sims = scalar_prods / norms
    assert torch.allclose(sims.sum(-1), torch.ones_like(sims.sum(-1)))
    return sims

def l2(decomposition, total_embs=None):
    if total_embs is None:
        total_embs = decomposition.sum(-2)
    total_embs = total_embs.unsqueeze(-2)
    return torch.linalg.norm(decomposition - total_embs, dim=-1)


def cosine(decomposition, total_embs=None):
    if total_embs is None:
        total_embs = decomposition.sum(-2)
    tgt_norms = torch.linalg.norm(total_embs, dim=-1, keepdims=True)
    dcp_norms = torch.linalg.norm(decomposition, dim=-1)
    scalar_prods = torch.einsum('lstd,lsd->lst', decomposition, total_embs)
    norms = tgt_norms * dcp_norms
    return scalar_prods / norms

def norm_ratio(decomposition, total_embs=None):
    if total_embs is None:
        total_embs = decomposition.sum(-2)
    tgt_norms = torch.linalg.norm(total_embs, dim=-1, keepdims=True)
    dcp_norms = torch.linalg.norm(decomposition, dim=-1)
    return dcp_norms / tgt_norms

def spims_from_files(decomposer, source_file, target_file, dump_file):
    with open(source_file) as src, open(target_file) as tgt, open(dump_file, 'w') as ostr:
        src, tgt = src.readlines(), tgt.readlines()
        assert len(src) == len(tgt)
        writer = csv.writer(ostr)
        _ = writer.writerow(['tok_idx', 'layer_idx', 'I', 'S', 'T', 'F', 'C', 'func', 'source', 'target'])
        for source, target in tqdm.tqdm(zip(src, tgt), total=len(src), desc='dcp'):
            source, target = source.strip(), target.strip()
            decomposition = decomposer(source, target, last_layer_only=False)
            total_embs = decomposition.sum(-2)
            for sim_func in spim, cosine, l2, norm_ratio:
                sims = sim_func(decomposition, total_embs=total_embs)
                assert sims.size() == decomposition.size()[:-1]
                for layer_idx in range(sims.size(0)):
                    for tok_idx in range(sims.size(1)):
                        I, S, T, F, C = sims[layer_idx,tok_idx].tolist()
                        _ = writer.writerow([tok_idx, layer_idx, I, S, T, F, C, sim_func.__name__, source, target])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('parallel corpus to CSV of spim')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--src', type=pathlib.Path, required=True)
    parser.add_argument('--tgt', type=pathlib.Path, required=True)
    parser.add_argument('--csv', type=pathlib.Path, required=True)
    parser.add_argument('--do_generate', action='store_true')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))
    args = parser.parse_args()
    if args.do_generate:
        model_mt = hf.MarianMTModel.from_pretrained(args.model).to(device=args.device, dtype=torch.float64)
        model_mt.eval()
        tokenizer = hf.MarianTokenizer.from_pretrained(args.model)
        with open(args.src) as istr:
            source_sentences = list(map(str.strip, istr))
        with open(args.tgt, 'w') as ostr:
            for source in tqdm.tqdm(source_sentences, desc='gen'):
                inputs = tokenizer(source, return_tensors='pt').to(device=args.device)
                translation = model_mt.generate(**inputs)
                print(tokenizer.decode(translation[0], skip_special_tokens=True), file=ostr)
        del model_mt, tokenizer
    decomposer = Decomposer(args.model, device=args.device)
    spims_from_files(decomposer, args.src, args.tgt, args.csv)
