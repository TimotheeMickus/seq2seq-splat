import pathlib
import pandas as pd

with open('/scratch/project_2005099/data/tatoeba/test/test.rus-eng.rus.txt') as istr:
    src = pd.Series(map(str.strip, istr))

with open('/scratch/project_2005099/data/tatoeba/test/test.rus-eng.eng.txt') as istr:
    tgt = pd.Series(map(str.strip, istr))

samples_per_seed = {}

for directory in pathlib.Path('sentence-evals/comet').glob('*'):
    print(directory.name)
    all_scores = []
    for file in directory.glob('*'):
        df = pd.read_csv(file, header=None, names=['ckpt', 'sentence_id', 'comet_score'])
        df['src'] = src
        df['tgt'] = tgt
        with open(df.ckpt.loc[0], 'r') as istr:
            df['hyp'] = pd.Series(map(str.strip, istr))
        all_scores.append(df)
    all_scores = pd.concat(all_scores)
    quantiles = all_scores.comet_score.quantile(q=[.25, .5, .75]).tolist()
    sub_samples = []
    for q_a, q_b in zip([0.] + quantiles, quantiles + [1.01]):
        subset = all_scores[(all_scores.comet_score >= q_a)  & (all_scores.comet_score <= q_b)]
        sub_samples.append(subset.sample(100))
    samples_per_seed[directory.name] = pd.concat(sub_samples).sort_values(by='ckpt').reset_index(drop=True)

for k, v in samples_per_seed.items():
    v.to_csv(f'{k}.csv', index=False)
    for ckpt, ckpt_df in v.groupby('ckpt'):
        the_iter = ckpt.split('.')[1].split('-')[0][len('iter'):]
        with open(f'sentence-evals/to-decompose/{k}.{the_iter}.src', 'w') as ostr:
             print(*ckpt_df.src.tolist(), sep='\n', file=ostr)
        with open(f'sentence-evals/to-decompose/{k}.{the_iter}.tgt', 'w') as ostr:
             print(*ckpt_df.hyp.tolist(), sep='\n', file=ostr)
