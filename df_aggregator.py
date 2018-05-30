#!/usr/bin/env

import pandas as pd
# it is better to rewrite in arrays, not dataframes


# params = {df=df, groups=[], pars=[],funcs=[],rename=None}
def aggregator(params):
    def func_parcer(df, gr, par, func):
        if func == 'count':
            return (df.groupby(gr)[par].count()).reset_index()
        elif func == 'mean':
            return (df.groupby(gr)[par].mean()).reset_index()
        elif func == 'min':
            return (df.groupby(gr)[par].min()).reset_index()
        elif func == 'max':
            return (df.groupby(gr)[par].max()).reset_index()
        elif func == 'unique_cnt':
            return (df.groupby(gr)[par].apply(lambda x: len(x.unique()))).reset_index()
        elif func == 'unique_sh':
            return (df.groupby(gr)[par].apply(lambda x: x.unique()[0])).reset_index()
        elif func == 'sum':
            return (df.groupby(gr)[par].sum()).reset_index()

    df = params['df']
    groups = params['groups']
    pars = params['pars']
    funcs = params['funcs']
    rename = params['rename']
    categorizer = None
    if 'categorizer' in params:
        categorizer = params['categorizer']
    tmp = df[:]
    for gr, par, func in zip(groups, pars, funcs):
        # print gr, par, func
        tmp = func_parcer(tmp, gr, par, func)
    if rename is not None:
        name = (set(pars)).pop()
        tmp = tmp.rename(columns={name: rename})
    if categorizer is not None:
        final = None

        value = list(tmp.columns).pop()
        key = list(tmp.columns).pop(0)

        cats = tmp[categorizer].unique()
        cols = list(tmp.columns)
        cols.remove(categorizer)
        if 'min_group_volume' in params:
            shape_norma = params['min_group_volume']
        else:
            shape_norma = max(500, int(0.01 * len(tmp[key].unique())))
        print shape_norma
        for cat in cats:
            d = tmp[tmp[categorizer] == cat][cols]
            if len(d[key].unique()) > shape_norma:
                d = d.rename(columns={value: "%s_%s" % (cat, value)})
                if final is None:
                    final = d[:]
                else:
                    final = pd.merge(final, d, how='outer', on=[key])
        final = pd.merge(final, tmp[[key]].drop_duplicates(), how='outer', on=[key])
        tmp = final.fillna(0)

    return tmp


#def periodations(func, pars):
#    dfs = []
#    global_df = pars['df'][:]
#    for n, i in enumerate(xrange(len(sorted_periods) - 2)):
#        pars['df'] = global_df[global_df['month_year'].isin(sorted_periods[i:i + 3])]
#        dfs.append(func(pars))
#    return dfs


def merger(dfs, field, how, fillna=None):
    new = None
    for df in dfs:
        if new is None:
            new = df[:]
        else:
            new = pd.merge(new, df, how=how, on=field)
    if fillna is not None:
        new = new.fillna(0)
    return new