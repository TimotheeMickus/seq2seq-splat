import warnings; warnings.filterwarnings('ignore')
import tqdm
import pandas as pd
import pathlib
import scipy.stats


#files = [x for x in pathlib.Path('results').glob('**/*-gen.csv') if int(x.name.split('-')[0][len('model.iter'):]) <= 585_000]
files_seed1 = sorted([x for x in pathlib.Path('results').glob('**/*.csv') if '-seed1' in x.name], key=lambda f: int(f.name.split('-')[0][len('model.iter'):]))
files_seed2 = sorted([x for x in pathlib.Path('results').glob('**/*.csv') if '-seed2' in x.name], key=lambda f: int(f.name.split('-')[0][len('model.iter'):]))
records = []
for file1, file2 in tqdm.tqdm(zip(files_seed1, files_seed2), total=min(len(files_seed1), len(files_seed2) ) ):
    df1 = pd.read_csv(file1).sort_values(by=['func', 'layer_idx', 'source', 'target', 'tok_idx'])
    df2 = pd.read_csv(file2).sort_values(by=['func', 'layer_idx', 'source', 'target', 'tok_idx'])
    ckpt = int(file1.name.split('-')[0][len('model.iter'):])
    for layer_idx in df1.layer_idx.unique():
        layer_df1 = df1[df1.layer_idx == layer_idx]
        layer_df2 = df2[df2.layer_idx == layer_idx]
        for func in df2.func.unique():
            func_df1 = layer_df1[layer_df1.func == func]
            func_df2 = layer_df2[layer_df2.func == func]
            for term in 'ISTFC':
                seed1_obs = func_df1[term].to_numpy()
                seed2_obs = func_df2[term].to_numpy()
                rho, pval = scipy.stats.spearmanr(seed1_obs, seed2_obs)
                records.append({'layer_idx': layer_idx, 'func': func, 'term': term, 'rho': rho, 'pval': pval, 'checkpoint': ckpt})



pd.DataFrame.from_records(records).to_csv('res-no-gen-correl-s1-s2.csv', index=False)
