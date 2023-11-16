import pandas as pd
import scipy.stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gen_df', type=pd.read_csv)
parser.add_argument('nogen_df', type=pd.read_csv)
parser.add_argument('--funcs', choices=['spim', 'cosine', 'norm_ratio'], nargs='+')
parser.add_argument('--terms', choices=['I', 'S', 'T', 'F', 'C', ], nargs='+')

correl_fn = {
    'spearman': scipy.stats.spearmanr,
    'pearson': scipy.stats.pearsonr,
}

parser.add_argument('--correl_fn', choices=correl_fn.keys(), default='spearman')
parser.add_argument('--prefix', default='')
parser.add_argument('--diffs', type=int, default=0)
args = parser.parse_args()

args.gen_df = args.gen_df[args.gen_df.layer_idx == 6]
args.nogen_df = args.nogen_df[args.nogen_df.layer_idx == 6]
if len(args.gen_df) != len(args.nogen_df):
    print(f'[WARNING] {len(args.gen_df)} != {len(args.nogen_df)}')
    valid_ckpts = set(min(args.gen_df, args.nogen_df, key=len).checkpoint.tolist())
    args.gen_df = args.gen_df[args.gen_df.checkpoint.apply(valid_ckpts.__contains__)]
    args.nogen_df = args.nogen_df[args.nogen_df.checkpoint.apply(valid_ckpts.__contains__)]

for term in args.terms:
    for func in args.funcs:
        gen_series = args.gen_df[args.gen_df.func == func].sort_values(by='checkpoint')[f'mean {term}']
        nogen_series = args.nogen_df[args.nogen_df.func == func].sort_values(by='checkpoint')[f'mean {term}']
        gen_series = gen_series.tolist()
        nogen_series = nogen_series.tolist()
        for _ in range(0, args.diffs):
            gen_series = [(t2 - t1) for t1, t2 in zip(gen_series, gen_series[1:])]
            nogen_series = [(t2 - t1) for t1, t2 in zip(nogen_series, nogen_series[1:])]
        print(args.prefix, func, term, *correl_fn[args.correl_fn](gen_series, nogen_series), sep=' & ', end='\\\\\n')
