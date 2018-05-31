#!/usr/bin/env

import pandas as pd
# it is better to rewrite in arrays, not dataframes

# Function for preparing aggregations out of dataframe
# depending on input parameters there are options for aggregation: group by and calculate one statistic
# OR calculate statistic depending on categorial feature
#
# Input format:
#
# params - dict-object, requirable fields: 'df', 'groups', 'pars', 'funcs'
#                       optional fields: 'rename', 'categorizer', 'min_group_volume'
#
# 'df' - supposes dataframe-value
# 'groups' - list of lists, each of them contains categories for group-by
# 'pars' - list of lists, each of them contains features to calculate aggregation function
#          (type - depends on aggregation function)
#
# 'funcs' - list of function-labels, available functions:
#
# - 'count'- calculate amount values
# - 'mean' - calculate mean value for feature
# - 'min' - calculate minimum of feature values
# - 'max' - calculate maximum of feature values
# - 'unique_cnt' - calculate amount of unique feature values
# - 'unique_sh' - returns one unique feature values
# - 'sum' - calculate sum of feature values
#
# 'rename' - name for new aggregated feature
# 'categorizer' - name of feature to split dataframe and calculate aggregation function through these categories
# 'min_group_volume' - integer, may be assigned while separating dataframe on categorial feature.
#                      Integer gives restriction on minimal sub-dataframe shape (separated by categorial featur values)
#
# Output format:
# dataframe aggregated according to input rule
#

def aggregator(params):
    # params = {df=df, groups=[], pars=[],funcs=[],rename=None}
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
        #print shape_norma
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


#def periodations(func, pars, sorted_periods, step):
#    dfs = []
#    global_df = pars['df'][:]
#    for n, i in enumerate(xrange(len(sorted_periods) - step +1)):
#        pars['df'] = global_df[global_df['month_year'].isin(sorted_periods[i:i + step])]
#        dfs.append(func(pars))
#    return dfs


# function merges several dataframes into one
#
# Input format:
#
# dfs - list of dataframes
# field - list of key-field/s to merge dataframes. This field/s must exist in all merging dataframes
# how - label, presents merging dataframes rule ('inner', 'outer', 'left', 'right')
# fillna - oprional parameter, if True, than in final dataframe all empry values will be filled with 0
#
# Output format:
#
# final merged dataframe
#

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