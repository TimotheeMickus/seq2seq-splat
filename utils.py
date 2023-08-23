import pandas as pd
import itertools
from dtw import dtw,accelerated_dtw


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
                if coso.get(comp) is not None:
                    cosonorm[comp]=(coso[comp].mean() - coso[comp])/coso[comp].std()
            
            normalized = pd.concat([normalized, pd.DataFrame(cosonorm)], ignore_index=True)
    return normalized

def read_datasets(dataset_name='both', multilingual=False,oh_decomp=False):
    """ Returns a dictionary with the MEANS of the read data structures. """
    decompname="-oh-schuler" if oh_decomp else ""
    auxstr=".rus-eng" if oh_decomp else ""
    components = ['mean I','mean S','mean T','mean F','mean C'] if not(oh_decomp) else ['mean S','mean T','mean C']
    newcols = {x:x.lstrip('mean ') for x in components}
    readitems=['layer_idx', 'func', 'checkpoint']+components
    datadict={x:{'gen':[],'non-gen':[]} for x in ['s0','s1','s2','sla','ine','mul']}
    if dataset_name=='gen':
        generative=True
        datadir=f"results/decomps{decompname}/gen"
        # compare the seeds of the gen decompositions
        s0=pd.read_csv(f'{datadir}/res{decompname}-gen.rus-eng-s0.csv')
        s1=pd.read_csv(f'{datadir}/res{decompname}-gen.rus-eng-s1.csv')
        s2=pd.read_csv(f'{datadir}/res{decompname}-gen.rus-eng-s2.csv')
        s0means = s0.filter(items=readitems, axis=1).rename(columns=newcols)
        s1means = s1.filter(items=readitems, axis=1).rename(columns=newcols)
        s2means = s2.filter(items=readitems, axis=1).rename(columns=newcols)
        if multilingual:
            sla=pd.read_csv(f'{datadir}/res{decompname}-gen.sla-eng.csv')
            ine=pd.read_csv(f'{datadir}/res{decompname}-gen.ine-eng.csv')
            mul=pd.read_csv(f'{datadir}/res{decompname}-gen.mul-eng.csv')
            slameans = sla.filter(items=readitems, axis=1).rename(columns=newcols)
            inemeans = ine.filter(items=readitems, axis=1).rename(columns=newcols)
            mulmeans = mul.filter(items=readitems, axis=1).rename(columns=newcols)
    elif ((dataset_name=='no-gen') or (dataset_name=='no gen') or (dataset_name=='nogen')):
        generative=False
        datadir=f"results/decomps{decompname}/no-gen"
        # gompare the seeds of no-gen decompositions
        s0=pd.read_csv(f'{datadir}/res{decompname}-no-gen.rus-eng-s0.csv')
        #s0 = s0[s0.checkpoint <= 585000] # drop ckpts that are not form this model
        s1=pd.read_csv(f'{datadir}/res{decompname}-no-gen.rus-eng-s1.csv')
        s2=pd.read_csv(f'{datadir}/res{decompname}-no-gen.rus-eng-s2.csv')
    elif ((dataset_name=='both') or (dataset_name=='gen-nogen')):
        generative='both'
        datadir=f"results/decomps{decompname}"
        # compare gen vs no-gen series
        s0=pd.read_csv(f'{datadir}/gen/res{decompname}-gen.rus-eng-s0.csv')
        s1=pd.read_csv(f'{datadir}/gen/res{decompname}-gen.rus-eng-s1.csv')
        s2=pd.read_csv(f'{datadir}/gen/res{decompname}-gen.rus-eng-s2.csv')
        ng0=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.rus-eng-s0.csv')
        ng1=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.rus-eng-s1.csv')
        ng2=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.rus-eng-s2.csv')
        s0means = ng0.filter(items=readitems, axis=1).rename(columns=newcols)
        s1means = ng1.filter(items=readitems, axis=1).rename(columns=newcols)
        s2means = ng2.filter(items=readitems, axis=1).rename(columns=newcols)
        ngs0means = ng0.filter(items=readitems, axis=1).rename(columns=newcols)
        ngs1means = ng1.filter(items=readitems, axis=1).rename(columns=newcols)
        ngs2means = ng2.filter(items=readitems, axis=1).rename(columns=newcols)
        if multilingual:
            sla = pd.read_csv(f'{datadir}/gen/res{decompname}-gen.sla-eng.csv')
            ine = pd.read_csv(f'{datadir}/gen/res{decompname}-gen.ine-eng.csv')
            mul = pd.read_csv(f'{datadir}/gen/res{decompname}-gen.mul-eng.csv')
            ngsla=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.sla-eng.csv')
            ngine=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.ine-eng.csv')
            ngmul=pd.read_csv(f'{datadir}/no-gen/res{decompname}-no-gen.mul-eng.csv')
            slameans = sla.filter(items=readitems, axis=1).rename(columns=newcols)
            inemeans = ine.filter(items=readitems, axis=1).rename(columns=newcols)
            mulmeans = mul.filter(items=readitems, axis=1).rename(columns=newcols)
            ngslameans = ngsla.filter(items=readitems, axis=1).rename(columns=newcols)
            nginemeans = ngine.filter(items=readitems, axis=1).rename(columns=newcols)
            ngmulmeans = ngmul.filter(items=readitems, axis=1).rename(columns=newcols)

    # LEGACY:
    #normalize=True
    #ZN=""
    #if normalize:
    #    ZN=" normalized"
    s0means = Znormalize(s0means)
    s1means = Znormalize(s1means)
    s2means = Znormalize(s2means)
    if generative =='both':
        ngs0means = Znormalize(ngs0means)
        ngs1means = Znormalize(ngs1means)
        ngs2means = Znormalize(ngs2means)
    if multilingual:
        slameans = Znormalize(slameans)
        inemeans = Znormalize(inemeans)
        mulmeans = Znormalize(mulmeans)
        if generative =='both':
            ngslameans = Znormalize(ngslameans)
            nginemeans = Znormalize(nginemeans)
            ngmulmeans = Znormalize(ngmulmeans)

    for name in ['s0','s1','s2','sla','ine','mul']:
        datadict[name]['gen'] = eval(name+'means')
        if generative=='both':
            datadict[name]['non-gen'] = eval('ng'+name+'means')

    if multilingual:
        if generative:
            maxckpt = min(datadict['s0']['gen'].checkpoint.max(),
                          datadict['s1']['gen'].checkpoint.max(),
                          datadict['s2']['gen'].checkpoint.max(),
                          datadict['sla']['gen'].checkpoint.max(),
                          datadict['ine']['gen'].checkpoint.max(),
                          datadict['mul']['gen'].checkpoint.max(),)
        else:
            maxckpt = min(datadict['s0']['gen'].checkpoint.max(),
                          datadict['s1']['gen'].checkpoint.max(),
                          datadict['s2']['gen'].checkpoint.max(),)
    metadata={'maxckpt':maxckpt, 'generative':generative}
    return metadata,datadict


def get_dtwdistances(models:dict(), distances:pd.DataFrame(), metric:str(), comp:str(), multilingual=False,doplots=False,decoding='gen'):

    nlayers=7
    if doplots:
        fig, ax = plt.subplots(nrows=len(models), ncols=nlayers, num=f'DTW{auxIDX} metric{metric} component{comp}')
        fig.suptitle(f'DTW Minimum Path comparison across seeds \n metric:{metric} and component:{comp}')

    #for i, m in enumerate(comparisons):
    for i,(m1,m2)in enumerate(itertools.combinations(models.keys(),2)):
        mod1, mname1 = models[m1][decoding], f'{m1}-eng'
        mod2, mname2 = models[m2][decoding], f'{m2}-eng'
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
            if doplots:
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
        distances = pd.concat([distances, pd.DataFrame([dlist], columns=distances.columns)])
    return distances