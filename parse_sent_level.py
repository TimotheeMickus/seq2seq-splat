import functools
import pandas as pd
import tqdm

@functools.lru_cache()
def get_decomps_df(seed, ckpt, oh_schuler=False):
    fname = f'sentence-evals/to-decompose/{seed}.{ckpt}{".oh-schuler" if oh_schuler else ""}.csv'
    return pd.read_csv(fname)


for seed in tqdm.tqdm(['rus-eng', 'rus-eng_s1989', 'rus-eng_s20232', 'sla-eng', 'ine-eng', 'mul-eng'], desc='seeds'):

    df = pd.read_csv(f'{seed}.csv')
    df['ckpt'] = df.ckpt.apply(lambda s: s.split('.')[1].split('-')[0][len('iter'):])
    df['seed'] = seed

    for oh_schuler in tqdm.tqdm([True, False]):
        def get_decompositions(row):
            seed, ckpt = row['seed'], row['ckpt']
            relevant_decomps = get_decomps_df(seed, ckpt, oh_schuler=oh_schuler)
            relevant_decomps =  relevant_decomps[(relevant_decomps.source == row['src']) & (relevant_decomps.target == row['hyp'])]
            new_subdf = relevant_decomps.groupby(by=['func', 'layer_idx']).mean(numeric_only=True).drop(columns='tok_idx').reset_index()
            for key_to_transfer in ['seed', 'ckpt', 'src', 'tgt', 'hyp', 'comet_score']:
	            new_subdf[key_to_transfer] = row[key_to_transfer]
            return new_subdf

        out_fname = f'res-sentence-level.{seed}{".oh-schuler" if oh_schuler else ""}.csv'
        pd.concat(df.apply(get_decompositions, axis=1).tolist()).to_csv(out_fname, index=False)

