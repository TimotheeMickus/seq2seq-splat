# import warnings; warnings.filterwarnings('ignore')
import argparse
import collections

import numpy as np
import tqdm
import pandas as pd
import pathlib
import scipy.stats
import multiprocessing as mp

p = argparse.ArgumentParser()
p.add_argument('--outname', type=str, required=True)
p.add_argument('ckpts', type=int, nargs='*')
p.add_argument('--dir1', type=pathlib.Path)
p.add_argument('--dir2', type=pathlib.Path)
args = p.parse_args()
args.ckpts = set(args.ckpts)

def get_ckpt(x):
    return int(x.with_suffix('').name.split('-')[0][len('model.iter'):])

#files = [x for x in pathlib.Path('results').glob('**/*-gen.csv') if int(x.name.split('-')[0][len('model.iter'):]) <= 585_000]
if args.ckpts: 
    test_func = lambda x: get_ckpt(x) in args.ckpts
else:
    test_func = lambda x: True
files_seed1 = sorted(
    [x for x in pathlib.Path(args.dir1).glob('**/*.csv') if test_func(x)],
    key=get_ckpt,
)
files_seed2 = sorted(
    [x for x in pathlib.Path(args.dir2).glob('**/*.csv') if test_func(x)],
    key=get_ckpt,
)

def process(packed):
    obs, key = packed
    seed1_obs, seed2_obs = obs
    u, pval = scipy.stats.mannwhitneyu(seed1_obs, seed2_obs)
    f = u / (len(seed1_obs) * len(seed2_obs))  # common language effect size, in [0 .. 1], 0.5 means same
    same = 1  - abs(2 * f - 1)  # 2 * f moves to 0 .. 2, (with 1 same), -1 moves to [-1 .. 1] (with 0 same), abs move to 0 .. 1 (with 0 same) 1 - moves to 0 .. 1 (with 1 same)
    computed = (u, pval, f, same)
    return (key, computed)


def yield_terms():
  for file1, file2 in tqdm.tqdm(zip(files_seed1, files_seed2), total=min(len(files_seed1), len(files_seed2) ) ):
    df = pd.concat([
        pd.read_csv(file1).rename(columns={t: f'{t} s1' for t in 'ISTFC'}),
        pd.read_csv(file2).rename(columns={t: f'{t} s2' for t in 'ISTFC'}),
    ])
    ckpt = get_ckpt(file1)
    assert ckpt == get_ckpt(file2)
    n_groups = len(df.source.unique()) * len(df.func.unique()) * len(df.layer_idx.unique())
    for (sentence, func, layer_idx), subdf in tqdm.tqdm(df.groupby(by=['source', 'func', 'layer_idx']), desc='config', leave=False, total=n_groups):
        for term in 'ISTFC':
            key = (layer_idx, func, term, ckpt)
            obs = (subdf[f'{term} s1'].dropna(), subdf[f'{term} s2'].dropna())
            if all(len(o) != 0 for o in obs):
                yield obs, key

stats = collections.defaultdict(list)
with mp.Pool(mp.cpu_count()) as pool:
    for pack in pool.imap_unordered(process, yield_terms(), 100):
        k, v = pack
        stats[k].append(v)

records = []
for (layer_idx, func, term, ckpt), vals in stats.items():
    us, ps, fs, sames = zip(*vals)
    records.append({'layer_idx': layer_idx, 'func': func, 'term': term, 'u (avg)': np.mean(us), 'pval (avg)': np.mean(ps), 'f (avg)': np.mean(fs), 'same': np.mean(sames), 'checkpoint': ckpt})

pd.DataFrame.from_records(records).to_csv(args.outname, index=False)
