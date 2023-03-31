import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw,accelerated_dtw


def plot_series_by_ind_and_layer(s, seed, functions:list()=None):
    '''
    Plot a time series of each component at a layer level
    '''
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
        fig, ax = plt.subplots(nrows=len(components), ncols=nlayers, num=f'scatter{auxIDX} metric:{f}')
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




def plot_dtw_allseeds(models:dict(), distances:pd.DataFrame(), metric:str(), comp:str(), multilingual=False):
    nlayers=7
    #fig, ax = plt.subplots(nrows=len(models), ncols=nlayers, num=f'DTW{auxIDX} metric{metric} component{comp}')
    #fig.suptitle(f'DTW Minimum Path comparison across seeds \n metric:{metric} and component:{comp}')

    comparisons = ['m1-m2', 'm2-m3','m3-m1'] if not multilingual else [
                   'm1-m2', 'm2-m3','m3-m1', 'ine-m1', 'ine-m2','ine-m3', 
                   'sla-m1', 'sla-m2','sla-m3', 'mul-m1', 'mul-m2','mul-m3',
                   ]
    for i, m in enumerate(comparisons):
        m1,m2 = m.split('-')
        mod1, mname1 = models[m1]['df'], models[m1]['modname']
        mod2, mname2 = models[m2]['df'], models[m2]['modname']
        df =  pd.DataFrame(mod1[mod1['func']==metric].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df = df.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}')
        df2 = pd.DataFrame(mod2[mod2['func']==metric].filter(['layer_idx',f'{comp}','checkpoint'])).reset_index(drop=True)
        df2 = df2.pivot(index='checkpoint', columns='layer_idx',values=f'{comp}') 
        dlist=[mname1,mname2,metric,comp]
        for j in range(nlayers):
            d1 = df[j].values
            d2 = df2[j].values
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1,d2, dist='euclidean')
            dlist.append(d)
            if False:
                currax = ax[i][j]
                currax.imshow(acc_cost_matrix.T, origin='lower', interpolation='nearest')
                currax.plot(path[0], path[1], 'w')
                if j == 0:
                    currax.set_xlabel(f'{mname1}', weight='bold')
                    currax.set_ylabel(f'{mname2}', weight='bold')
                if i == 0:
                    currax.set_title(f'layer {j}')
                    currax.text(50, 500, f'distance: {np.round(d,3)}', 
                     bbox=dict(fill=True, color='gray'))
            
        distances = distances.append(pd.DataFrame([dlist], columns=distances.columns))
    return distances




sys.argv = sys.argv if len(sys.argv)>=2 else [sys.argv[0],'gen','']
def main():
    if sys.argv[1]=='gen':
        # compare the seeds of the gen decompositions
        s0=pd.read_csv('results/decomps/gen/res-gen-s0.csv')
        s1=pd.read_csv('results/decomps/gen/res-gen-s1.csv')
        s2=pd.read_csv('results/decomps/gen/res-gen-s2.csv')
        s0means = s0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s1means = s1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s2means = s2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        if sys.argv[2]=='multilingual':
            sla=pd.read_csv('results/decomps/gen/res-gen.sla-eng.csv')
            ine=pd.read_csv('results/decomps/gen/res-gen.ine-eng.csv')
            mul=pd.read_csv('results/decomps/gen/res-gen.mul-eng.csv')
            slameans = sla.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            inemeans = ine.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
            mulmeans = mul.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
    elif ((sys.argv[1]=='no-gen') or (sys.argv[1]=='no gen') or (sys.argv[1]=='nogen')):
        # gompare the seeds of no-gen decompositions
        s0=pd.read_csv('results/decomps/no-gen/res-no-gen-s0.csv')
        s0 = s0[s0.checkpoint <= 585000] # drop ckpts that are not form this model
        s1=pd.read_csv('results/decomps/no-gen/res-no-gen-s1.csv')
        s2=pd.read_csv('results/decomps/no-gen/res-no-gen-s2.csv')
    elif ((sys.argv[1]=='both') or (sys.argv[1]=='gen-nogen')):
        # compare gen vs no-gen series
        s0=pd.read_csv('results/decomps/gen/res-gen-s0.csv')
        s1=pd.read_csv('results/decomps/gen/res-gen-s1.csv')
        s2=pd.read_csv('results/decomps/gen/res-gen-s2.csv')
        ng0=pd.read_csv('results/decomps/no-gen/res-no-gen-s0.csv')
        ng1=pd.read_csv('results/decomps/no-gen/res-no-gen-s1.csv')
        ng2=pd.read_csv('results/decomps/no-gen/res-no-gen-s2.csv')
        s0means = ng0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s1means = ng1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        s2means = ng2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ng0means = g0.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ng1means = g1.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})
        ng2means = g2.filter(items=['layer_idx', 'func', 'mean I', 'mean S', 'mean T', 'mean F', 'mean C', 'checkpoint'], axis=1).rename(columns={'mean I':'I', 'mean S':'S', 'mean T':'T', 'mean F':'F', 'mean C':'C'})




    global auxIDX # aux in naming the Figure windows
    auxIDX = 1
    if not ((sys.argv[1]=='both') or (sys.argv[1]=='gen-nogen')):
        plot_series_by_ind_and_layer(s0means,seed=1111)
        auxIDX += 1
        plot_series_by_ind_and_layer(s1means,seed=1989)
        auxIDX += 1
        plot_series_by_ind_and_layer(s2means,seed=20232)

        auxIDX = 1
        for func in s0means.func.unique():
            figure  = scatter_2seeds(s0means, '1111', s1means, '1988', func)
            scatter_2seeds(s0means, '1111', s2means, '20232', func, overlapfig=figure)
            scatter_2seeds(s1means, '1989', s2means, '20232', func, overlapfig=figure)
            auxIDX+=1
            #savefig... (f'RUS-ENG_SCATTERseeds_{func}.pdf') <- need the right size

        #   ATTEMPT: Dynamic Time Warp
        if sys.argv[2]=='multilingual':
            models  ={'m1': {'df':s0means.copy(), 'modname':'rus-eng s1'},
                  'm2': {'df':s1means.copy(), 'modname':'rus-eng s2'},
                  'm3': {'df':s2means.copy(), 'modname':'rus-eng s3'},
                  'sla': {'df':slameans.copy(), 'modname':'sla-eng'},
                  'ine': {'df':inemeans.copy(), 'modname':'ine-eng'},
                  'mul': {'df':mulmeans.copy(), 'modname':'mul-eng'},}
        else:
            models  ={'m1': {'df':s0means.copy(), 'modname':'rus-eng s1'},
                  'm2': {'df':s1means.copy(), 'modname':'rus-eng s2'},
                  'm3': {'df':s2means.copy(), 'modname':'rus-eng s3'},}

        dfdist = pd.DataFrame(columns=['model1','model2', 'metric','component', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6'])
        for func in s0means.func.unique():
            for comp in ['I', 'S', 'T', 'F', 'C']:
                print(func,comp)
                dfdist = plot_dtw_allseeds(models, dfdist, func, comp, sys.argv[2]=='multilingual')
                auxIDX+=1
        dfdist.to_csv('results/DTWdistances-res-gen.csv', index=False)
        import ipdb; ipdb.set_trace()
        
        plt.show()


    

if __name__ == '__main__':
    main()
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

