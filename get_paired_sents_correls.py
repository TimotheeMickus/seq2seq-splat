import pathlib

import numpy as np
import pandas as pd
import scipy.stats

pdir = pathlib.Path('results/sentence-level/paired-sents/')

keys, dfs = [], []
for df in map(pd.read_csv, pdir.glob('*.mickus+al.scored.csv')):
	for key, subdf in df[df.layer_idx == 6].groupby(['func', 'sentnum', 'seed']):
		keys.append(key)
		dfs.append(subdf.select_dtypes(include=np.number).diff().iloc[1:])

df = pd.concat(dfs)
func, sentnum, seed = zip(*keys)
df['func'] = func
df['sentnum'] = sentnum
df['seed'] = seed

testables = [col for col in df.columns if 'mean' in col]

for key, subdf in df.groupby(['func', 'seed']):
	print('%', *key)
	for testable in testables:
		print('\t(', testable.split()[1], ',', abs(scipy.stats.spearmanr(subdf[testable], subdf['scores'])[0]) * 100, ') %', scipy.stats.spearmanr(subdf[testable], subdf['scores'])[1])
	print()

keys, dfs = [], []
for df in map(pd.read_csv, pdir.glob('*.oh+schuler.scored.csv')):
	for key, subdf in df[df.layer_idx == 6].groupby(['func', 'sentnum', 'seed']):
		candidate = subdf.select_dtypes(include=np.number).diff().iloc[1:]
		if len(candidate):
			keys.append(key)
			dfs.append(candidate)

df = pd.concat(dfs)
func, sentnum, seed = zip(*keys)
df['func'] = func
df['sentnum'] = sentnum
df['seed'] = seed

testables = [col for col in df.columns if 'mean' in col]

for key, subdf in df.groupby(['func', 'seed']):
	print('%', *key)
	for testable in testables:
		print('\t(', testable.split()[1], ',', abs(scipy.stats.spearmanr(subdf[testable], subdf['scores'])[0]) * 100, ') %', scipy.stats.spearmanr(subdf[testable], subdf['scores'])[1])
	print()

