import argparse 
import tqdm
import pandas as pd
import pathlib

p = argparse.ArgumentParser()
p.add_argument('topdir', type=pathlib.Path)
p.add_argument('outfile', type=pathlib.Path)
p.add_argument('--oh-schuler', action='store_true')
args = p.parse_args()
topdir = args.topdir
outfile = args.outfile

#files = [x for x in pathlib.Path('results').glob('**/*-gen.csv') if int(x.name.split('-')[0][len('model.iter'):]) <= 585_000]
files = [x for x in topdir.glob('**/*.csv')]
dfs=[]

cols = ['I', 'S', 'T', 'F', 'C']
if args.oh_schuler:
    cols = ['S', 'T', 'C']


for file in tqdm.tqdm(files):
    df = pd.read_csv(file)
    df_mean = df.groupby(by=['layer_idx', 'func'])[cols].mean().fillna(0.0).reset_index().rename(columns={t: 'mean ' + t for t in cols})
    df_std = df.groupby(by=['layer_idx', 'func'])[cols].std().fillna(0.0).reset_index().rename(columns={t: 'std ' + t for t in cols})
    df = pd.merge(df_mean, df_std).sort_values(by=['layer_idx', 'func'])
    df['checkpoint'] = int(file.with_suffix("").name.split('-')[0][len('model.iter'):])
    dfs.append(df)


pd.concat(dfs).sort_values(by=['checkpoint', 'layer_idx', 'func']).to_csv(outfile, index=False)
