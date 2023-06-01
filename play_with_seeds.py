import sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw,accelerated_dtw
import scipy.stats as stats


def plot_series_by_ind_and_layer(s, seed, modname:str()='rus-eng', functions:list()=None):
    '''
    Plot a time series of each component at a layer level
    '''
    components= ['I', 'S', 'T', 'F', 'C']
    func = functions if functions else s.func.unique()
    fig, ax = plt.subplots(ncols=len(components), nrows=len(func), sharex=True, num=f'TS by layer - {modname} {seed}')
    fig.suptitle(f'MODEL: {modname}   {seed}')
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

        for j in range(1,nlayers):
            n = min(len(df[j]),len( df2[j]))

            currax= ax[i][j]
            #dots = currax.scatter(df[j][:n],df2[j][:n], alpha=0.5, label=f'seed {seed1} vs seed {seed2}')
            dots = currax.plot(df[j][:n].to_numpy(),df2[j][:n].to_numpy(), alpha=0.5, label=f'seed {seed1} vs seed {seed2}')
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

    comparisons = ['m1-m2',   'm2-m3',   'm3-m1'] if not multilingual else [
                   'm1-m2',   'm2-m3',   'm3-m1',  'ine-m1', 'ine-m2','ine-m3', 
                   'sla-m1',  'sla-m2',  'sla-m3', 'mul-m1', 'mul-m2','mul-m3',
                   'ine-sla', 'ine-mul', 'sla-mul',
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


def Znormalize(df:pd.DataFrame()):
    # z-normalize a given dataframe
    normalized = pd.DataFrame(columns=df.columns)
    for metric in df.func.unique():
        for layer in df.layer_idx.unique():
            coso = df[((df.func==metric)&(df.layer_idx==layer))]
            cosonorm = {'layer_idx': [layer]*len(coso), 
                      'func':[metric]*len(coso), 
                      'checkpoint':coso.checkpoint }
            for comp in ['I', 'S', 'T', 'F', 'C']:
                cosonorm[comp]=(coso[comp].mean() - coso[comp])/coso[comp].std()
            
            normalized = pd.concat([normalized, pd.DataFrame(cosonorm)], ignore_index=True)
    return normalized

def plotDTWheatmaps(df):
    fig, ax = plt.subplots(nrows=len(df.metric.unique()), ncols=3)
    for i,func in enumerate(df.metric.unique()):
        for j,seed in enumerate(['s1', 's2', 's3']):
            currax = ax[i][j]
            coso = df[((df.metric==func)  & ((df.model2==f'rus-eng {seed}') | (df.model1==f'rus-eng {seed}') ))]# & (df.component!='C')]
            currax.imshow(coso.iloc[:,-6:].T)
            #xlabels =  [f"{i} \t{j.split(' ')[-1].split('-')[0]}-{k.split(' ')[-1].split('-')[0]}".expandtabs() for i,j,k in zip(coso.component,coso.model1,coso.model2)]
            #xlabels = [f"{k} {l.split(' ')[-1].split('-')[0]}-{m.split(' ')[-1].split('-')[0]}" for k,l,m in zip(coso.component,coso.model1,coso.model2)]
            xlabels = [f"{k} {l.split(' ')[-1].split('-')[0]}" for k,l in zip(coso.component,coso.model1)]
            for idx,x in enumerate(xlabels):
                if x.find(seed) > 0 :
                    xlabels[idx] = x.split(' ')[0]+' s'+str((int(seed.lstrip('s'))+1)%4)
            currax.set_xticks([i for i in range(len(coso))])
            currax.set_xticklabels(['']*len(coso))
            #currax.set_yticklabels(["","2","","4","","6"])
            if j==0:
                currax.set_ylabel(func)
            if i==0:
                currax.set_title(f'DTW dist. to rus-eng seed {seed}')
            if i==(len(df.metric.unique())-1):
                #currax.set_yticks([1,2,3,4,5,6])
                #currax.set_yticklabels(["","2","","4","","6"])
                currax.set_xlabel('COMPONENT  distance-to-model')
                currax.set_xticklabels(xlabels)
                currax.xaxis.set_tick_params(rotation=90)

def do_tests(g1,g2,func,comp,seed=""):
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for layer in range(1,7):
        pop1, pop2 = g1[f'layer{layer}'], g2[f'layer{layer}']
        pop = pd.concat((pop1,pop2))
        appendable = (func,comp,layer,seed) if seed else (func,comp,layer)
        # U-test
        ustat = stats.mannwhitneyu(pop1, pop2, alternative='two-sided')
        meandiff = pop1.mean() - pop2.mean()
        if ustat.pvalue < 0.05:
            uH0reject.append(appendable)
        else:
            uH0notreject.append(appendable)
        # permutation test:
        mudiffs = []
        for part1 in itertools.combinations(pop.index, len(pop1)): 
            part2 = [x for x in pop.index if x not in part1]
            mudiffs.append(pop.loc[np.array(part1)].mean() - pop.loc[np.array(part2)].mean())
        ge = [1 for x in mudiffs if x > meandiff]
        pval = sum(ge)/len(mudiffs)
        if pval < 0.05:
            permH0reject.append(appendable)
        else:
            permH0notreject.append(appendable)
    return uH0reject, uH0notreject, permH0reject, permH0notreject

def permutation_and_U_tests(df):
    # PERFORM MannWhitney U-test to see if distance distributions are equivalent:
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE1: fix k, then g1={DTWdist(sk,sj) \forall seeds j } and g2={DTWdist(multilingual,sj) \forall seeds j}
            for seed in ['s1', 's2', 's3']:
                # there are 360cases here: 6layer*3seeds*5components*4metrics
                coso = df[((df.metric==func)  & ((df.model2==f'rus-eng {seed}') | (df.model1==f'rus-eng {seed}') )) & (df.component==comp)]
                ing1=coso.model1.str.contains(" s")
                g1 = coso[ing1]
                g2 = coso[ing1==0]
                tests = do_tests(g1,g2,func,comp,seed)
                uH0reject = uH0reject + tests[0]
                uH0notreject = uH0notreject + tests[1]
                permH0reject= permH0reject + tests[2]
                permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 1 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    #for rej in H0reject:
    #    print(rej)
    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE2: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(multilingual,si) \forall seed i}U{DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 2 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    #for rej in H0reject:
    #    print(rej)

    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE3: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(multilingual,si) \forall seed i}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains(" s")]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 3 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    #for rej in H0reject:
    #    print(rej)

    H0reject, H0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE4: g1={DTWdist(si,sj) \forall seeds i,j } and g2={DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s")) & (df.metric==func) & (df.component==comp)]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains('sla|ine|mul')]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 4 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    #for rej in H0reject:
    #    print(rej)

    uH0reject, uH0notreject=[],[]
    permH0reject, permH0notreject=[],[]
    for func in df.metric.unique():
        for comp in ['I', 'S', 'T', 'F', 'C']:
            # CASE5: g1={DTWdist(mi,mj) \forall mh \in SET={si \forall seed i}U{sla}} and g2={DTWdist(mult1,mult2) for multN in {mul,sla,ine}}
            g1 = df[(df.model1.str.contains(" s|sla")) & (df.metric==func) & (df.component==comp)]
            g1 = g1[g1.model2.str.contains(' s')]
            g2 = df[(df.model1.str.contains('sla|ine|mul')) & (df.metric==func) & (df.component==comp)]
            g2 = g2[g2.model2.str.contains('sla|ine|mul')]
            tests = do_tests(g1,g2,func,comp)
            uH0reject = uH0reject + tests[0]
            uH0notreject = uH0notreject + tests[1]
            permH0reject= permH0reject + tests[2]
            permH0notreject = permH0notreject + tests[3]

    #H0reject is EMPTY in this case -> there is no sufficient evidence to reject the H0: "the two samples come from the same distribution"
    print(f"CASE 5 -   Utest: the null hypothesis is rejected on {len(uH0reject)} our of the {len(uH0reject)+len(uH0notreject)} times. They are:")
    print(f"   - Permutation: the null hypothesis is rejected on {len(permH0reject)} our of the {len(permH0reject)+len(permH0notreject)} times. They are:")
    #for rej in H0reject:
    #    print(rej)



sys.argv = sys.argv if len(sys.argv)>=2 else [sys.argv[0],'gen','']
def main(multilingual=False):
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



    normalize=True
    ZN=""
    if normalize:
        ZN=" normalized"
        s0means = Znormalize(s0means)
        s1means = Znormalize(s1means)
        s2means = Znormalize(s2means)
        if multilingual:
            slameans = Znormalize(slameans)
            inemeans = Znormalize(slameans)
            mulmeans = Znormalize(slameans)
    global auxIDX # aux in naming the Figure windows
    auxIDX = 1
    if not ((sys.argv[1]=='both') or (sys.argv[1]=='gen-nogen')):
        if False:
            plot_series_by_ind_and_layer(s0means,seed=str(1111)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(s1means,seed=str(1989)+ZN)
            auxIDX += 1
            plot_series_by_ind_and_layer(s2means,seed=str(20232)+ZN)
            if multilingual:
                auxIDX += 1
                plot_series_by_ind_and_layer(slameans,seed=ZN, modname='sla-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(inemeans,seed=ZN, modname='ine-eng')
                auxIDX += 1
                plot_series_by_ind_and_layer(mulmeans,seed=ZN, modname='mul-eng')

            plt.show()
        auxIDX = 1
        for func in s0means.func.unique():
            figure  = scatter_2seeds(s0means, '1111', s1means, '1988', func)
            scatter_2seeds(s0means, '1111', s2means, '20232', func, overlapfig=figure)
            scatter_2seeds(s1means, '1989', s2means, '20232', func, overlapfig=figure)
            auxIDX+=1
            #savefig... (f'RUS-ENG_SCATTERseeds_{func}.pdf') <- need the right size

        #   ATTEMPT: Dynamic Time Warp
        if sys.argv[2]=='multilingual':
            dflen = min(len(s0means),len(s1means),len(s2means),len(slameans),len(inemeans),len(mulmeans))
            models  ={'m1': {'df':s0means[:dflen].copy(), 'modname':'rus-eng s1'},
                  'm2': {'df':s1means[:dflen].copy(), 'modname':'rus-eng s2'},
                  'm3': {'df':s2means[:dflen].copy(), 'modname':'rus-eng s3'},
                  'sla': {'df':slameans[:dflen].copy(), 'modname':'sla-eng'},
                  'ine': {'df':inemeans[:dflen].copy(), 'modname':'ine-eng'},
                  'mul': {'df':mulmeans[:dflen].copy(), 'modname':'mul-eng'},}
        else:
            dflen = min(len(s0means),len(s1means),len(s2means))
            models  ={'m1': {'df':s0means[:dflen].copy(), 'modname':'rus-eng s1'},
                  'm2': {'df':s1means[:dflen].copy(), 'modname':'rus-eng s2'},
                  'm3': {'df':s2means[:dflen].copy(), 'modname':'rus-eng s3'},}



        dfdist = pd.DataFrame(columns=['model1','model2', 'metric','component', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6'])
        nom = "_multilingual" if sys.argv[2]=='multilingual' else "_bilingual"
        if os.path.exists(f'results/DTWdistances-res-gen{nom}.csv'):
            dfdist = pd.read_csv(f'results/DTWdistances-res-gen{nom}.csv')
        else:
            for func in s0means.func.unique():
                for comp in ['I', 'S', 'T', 'F', 'C']:
                    print(func,comp)
                    dfdist = plot_dtw_allseeds(models, dfdist, func, comp, sys.argv[2]=='multilingual')
                    auxIDX+=1
            dfdist.to_csv(f'results/DTWdistances-res-gen{nom}.csv', index=False)


        # dfdist = pd.read
        df = dfdist.copy().reset_index(drop=True)
        plotDTWheatmaps(df)

        #plt.show()
        # PERFORM MannWhitney U-test to see if distance distributions are equivalent:
        permutation_and_U_tests(df)

        import ipdb; ipdb.set_trace()


    

if __name__ == '__main__':
    multilingual = True if sys.argv[2]=='multilingual' else False
    main(multilingual)
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

