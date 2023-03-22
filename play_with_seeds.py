import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s0=pd.read_csv('results/decomps/gen/res-gen-s0.csv')
s1=pd.read_csv('results/decomps/gen/res-gen-s1.csv')
s2=pd.read_csv('results/decomps/gen/res-gen-s2.csv')

s0means = s0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
s1means = s1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
s2means = s2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})

s0cos = s0means[s0['func']=='cosine']
s1cos = s1means[s1['func']=='cosine']
s2cos = s2means[s2['func']=='cosine']


"""
BROKEN... need to figure out how to compare each checkpoint
def make_scattermatrix(s:pd.DataFrame(), functions:list()=None):
    '''
    make animation of the scatter_matrix of the decomposed I,S,T,F,C
    '''
    func = functions if functions else s.func.unique() 
    for f in func:
        for ckpt in s.checkpoint.unique():
            pd.plotting.scatter_matrix(s[(s['checkpoint']==ckpt)&(s['func']==f)].filter(items=['I','S','T','F','C']) )
            plt.suptitle(f'{f} ckpt {ckpt}')
"""

global auxIDX
auxIDX=1
def plot_series_by_ind_and_layer(s, seed, functions:list()=None):
    components= ['I', 'S', 'T', 'F', 'C']
    func = functions if functions else s.func.unique()
    fig, ax = plt.subplots(ncols=len(components), nrows=len(func), sharex=True, num=f'TS by layer{auxIDX} seed{seed}')
    fig.suptitle(f'MODEL: rus-eng   SEED: {seed}')
    for j, comp in enumerate(components):
        for i,f in enumerate(func): 
            currax= ax[i][j]
            df =  pd.DataFrame(s[s['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
            df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')
            df.plot(ax=currax)
            currax.get_legend().remove()
            if j==0:
                currax.set_ylabel(f'{f}')
            if i==0:
                currax.set_title(f'{comp}')

    currax.legend(bbox_to_anchor=(1.64, 4),title='layer')

#plt.ion()
plot_series_by_ind_and_layer(s0means,seed=1111)
auxIDX +=1
plot_series_by_ind_and_layer(s1means,seed=1989)
auxIDX +=1
plot_series_by_ind_and_layer(s2means,seed=20232)



def scatter_2seeds(s1, seed1:str(), s2, seed2:str(), f:str(), overlapfig:tuple()=(None,None)):
    '''
    make scatter plots matrix comparing two seeds
    IN:
        s1: dataframe sImeans like
        seed1: string indicating the seed of s1
        s2: another dataframe
        seed2: string indicating the seed of s2
        f: either 'cosine', 'norm_ratio', 'l2' or 'spim'
    '''
    components= ['I', 'S', 'T', 'F', 'C']
    nlayers=7
    if overlapfig[0]:
        fig, ax = overlapfig
        fig.suptitle(f'Scatter plots of {f} for rus-eng models with all three seeds')

    else:
        fig, ax = plt.subplots(nrows=len(components), ncols=nlayers, num=f'scatter{auxIDX} metric{f}')
        fig.suptitle(f'Scatter of {f} for rus-eng models with seeds: {seed1} & {seed2}')

    for i,comp in enumerate(components):
        df =  pd.DataFrame(s1[s1['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')

        df2 = pd.DataFrame(s2[s2['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df2 = df2.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}') 

        for j in range(nlayers):
            n = min(len(df[j]),len( df2[j]))

            currax= ax[i][j]
            dots = currax.scatter(df[j][:n],df2[j][:n], alpha=0.5, label=f'seed {seed1} vs seed {seed2}')
            #currax.get_legend().remove()

            if j==0:
                currax.set_ylabel(f'{comp}', weight='bold')
            if i==0:
                currax.set_title(f'Layer {j}')
        
    currax.legend(bbox_to_anchor=(2, 3.5))
    
    return fig,ax

auxIDX=1
for func in s0means.func.unique():
    figure  = scatter_2seeds(s0means, '1111', s1means, '1988', func)
    scatter_2seeds(s0means, '1111', s2means, '20232', func, overlapfig=figure)
    scatter_2seeds(s1means, '1989', s2means, '20232', func, overlapfig=figure)
    auxIDX+=1
    #savefig... (f'RUS-ENG_SCATTERseeds_{func}.pdf') <- need the right size

#plt.show()



# Time Series
#   ATTEMPT: Dynamic Time Warp
from dtw import dtw,accelerated_dtw


f='spim'
comp='F'

def plot_dtw_allseeds(models:dict(), f):
    nlayers=7
    fig, ax = plt.subplots(nrows=3, ncols=nlayers, num=f'DTW{auxIDX} metric{f} component{comp}')
    fig.suptitle(f'DTW Minimum Path comparison across seeds \n metric:{f} and component:{comp}')

    for i, m in enumerate(['m1-m2', 'm2-m3','m3-m1']):
        m1,m2 = m.split('-')
        s1, seed1 = models[m1]['df'], models[m1]['seed']
        s2, seed2 = models[m2]['df'], models[m2]['seed']
        
        df =  pd.DataFrame(s1[s1['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')
        df2 = pd.DataFrame(s2[s2['func']==f].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df2 = df2.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}') 
        for j in range(nlayers):
            currax = ax[i][j]
            d1 = df[j].values
            d2 = df2[j].values
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1,d2, dist='euclidean')

            currax.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            currax.plot(path[0], path[1], 'w')
            if j == 0:
                currax.set_xlabel(f'seed{seed1}', weight='bold')
                currax.set_ylabel(f'seed{seed2}', weight='bold')
            if i == 0:
                currax.set_title(f'layer {j}')
            currax.text(50, 500, f'distance: {np.round(d,3)}', 
                     bbox=dict(fill=True, color='gray'))



models  ={'m1': {'df':s0means.copy(), 'seed':'1111'},
          'm2': {'df':s1means.copy(), 'seed':'1989'},
          'm3': {'df':s2means.copy(), 'seed':'20232'},}

auxIDX=0
for func in s0means.func.unique():
    plot_dtw_allseeds(models, func)
    auxIDX+=1
plt.show()

"""
FAILED ATTEMPT: Pearson correlation & rolling-window version of it... 
                failed bc the homoscedasticity assumption does not hold

Pearson correlation tests to measure how two continuous signals co-vary over time and 
indicate the linear relationship as a number between -1 (negatively correlated) to 0 (not correlated) to 1 (perfectly correlated).
ASSUMPTION:
     homoscedasticity of the data (i.e., equal variance throughout observations)

df = df =  pd.DataFrame(s[s['func']==f].filter(['layer_idx',f'{ind}','checkpoint'])).reset_index(drop=True)
df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{ind}')


# PART 1: is the homoscedasticity assumprion holds?
from sklearn.linear_model import LinearRegression

x,y = df.index.array.reshape(-1,1),df[6].array.reshape(-1,1)
regr = LinearRegression()
regr.fit(x,y)
y_hat= regr.predict(y)
residuals= y-y_hat
plt.scatter(y_hat,residuals)
# Breusch-Pagan test
residuals2 = residuals ** 2 # square the residuals
regr2 = LinearRegression()  # new regression model using the squared residuals as the response values
regr2.fit(x,residuals2)
# Calculate the Chi-Square test statistic X2 as num_observs * R2 (R2 is coef of determination of regr2)
r2 = regr2.score(x,residuals2) 
"""
