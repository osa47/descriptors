#!/usr/bin/env

from df_aggregator import aggregator
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import csr_matrix, csgraph
from itertools import combinations
from datetime import datetime as dt
from tqdm import tqdm
# it is better to rewrite in arrays, not dataframes

from sklearn import metrics


# Styler function for dataframes, marks font-red values for tests H0 rejections
# Used within dataframe.style.applymap(parameter) construction as parameter

def color_sign_red(val):
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color


# 2 groups metric feature comparison (parametric/non-parametric tests)
# Parametric - Student ttest if metric feature satisfies normal-distribution requirement,
# Non-parametric - Mann-Whitney U - if metric is not normally distributed
#
# Input format:
#
# df - dataframe object with fields f_name, groupper (default - "target", 2-values-variable {0,1})
# f_name - name of a metric variable
# groupper - name of a 2-values-variable, groupping
#
# Output format:
#
# groups_stats - dataframe with descriptive statistics on each group, divided with groupping variable,
# and a pvalue of testing H0 - if groups are out of the same distribution
# styler - transformed dataframe with styler "color_sign_red"

def pairwise_comparison(df, f_name, groupper='target'):
    stats_1 = df[df[groupper] == 1][f_name]
    stats_0 = df[df[groupper] == 0][f_name]
    statistic, pval = stats.normaltest(df[f_name].values)
    if pval > 0.05:
        statistic, pvalue = stats.ttest_ind(stats_1.values, stats_0.values)
        label = 'Student'
    else:
        statistic, pvalue = stats.mannwhitneyu(stats_1.values, stats_0.values)
        label = 'MannWhitneyU'

    groups_stats = [['Minimum', stats_1.min(), stats_0.min()],
                    ['Maximum', stats_1.max(), stats_0.max()],
                    ['Average', stats_1.mean(), stats_0.mean()],
                    ['Std.error', stats_1.std(), stats_0.std()],
                    ['%s_st_pvalue' % label, statistic, pvalue],
                    ['25prcnt', np.percentile(stats_1.values, 25), np.percentile(stats_0.values, 25)],
                    ['50prcnt', np.percentile(stats_1.values, 50), np.percentile(stats_0.values, 50)],
                    ['75prcnt', np.percentile(stats_1.values, 75), np.percentile(stats_0.values, 75)]]
    groups_stats = pd.DataFrame(groups_stats)
    groups_stats.columns = ['StatName', 'Group1', 'Group0']
    styler = groups_stats.style.applymap(color_sign_red, subset=pd.IndexSlice[4:4, ['Group0']])

    return groups_stats,styler,pvalue


# 2 metric features comparison (correlations - parametric/non-parametric tests)
# function calculates 3 tables:
# for metric-features of normal distribution - pearson coef (presenting H0 pvalue if coef==0)
# for metric-features not normal distributed - sperman coef (presenting H0 pvalue if coef==0)
# table of mutual information metric, styler presenting values above 2-sigmas
# three stylers for each table
#
# Input format:
#
# df - dataframe, containing features to estimate
# features - list of features to estimate (each with each)
#
# Output format:
# three tuples
# - first tuple of 2 lists: statistically non correlated according to Pearson test, statistically correlated
# - second tuple of 2 lists: statistically non correlated according to Spearman test, statistically correlated
# - third tuple of dataframe and styler for mutual information statistics
#

def correlations(df, features,crit_val=0.05):
    # Pearson correlation measures the linear association between continuous variables
    # Spearman's correlation measures monotonic association (only strictly increasing or decreasing,
    # but not mixed) between two variables and relies on the rank order of values.

    # The p-value roughly indicates the probability of an uncorrelated system producing datasets

    def color_with_criterio(val):
        color = 'red' if val >= criterio else 'black'
        return 'color: %s' % color

    combs =  combinations(features,2)

    mutuals = [] #np.zeros((len(features), len(features)))
    pearson_N = [] #np.zeros((len(features), len(features)))
    pearson_Y = []
    spearman_N = [] #np.zeros((len(features), len(features)))
    spearman_Y=[]

    for comb in combs:
        pearson_flag = False
        _, pval = stats.normaltest(df[comb[0]].values)
        if pval > 0.05:
            pearson_flag = True
        _, pval = stats.normaltest(df[comb[1]].values)
        if ((pval > 0.05) and (pearson_flag)):
            pearson_flag = True
            pear, pearval = stats.pearsonr(df[comb[0]].values, df[comb[1]].values)
            if pearval < 0.05:
                pearson_Y.append([comb[0],comb[1], pearval])
            else:
                pearson_N.append([comb[0], comb[1], pearval])

        spear, spearval = stats.spearmanr(df[comb[0]].values, df[comb[1]].values)
        if spearval<crit_val:
            spearman_Y.append([comb[0], comb[1], spearval])
        else:
            spearman_N.append([comb[0], comb[1], spearval])
        mutuals.append([comb[0], comb[1], metrics.mutual_info_score(((df[comb[0]] - df[comb[0]].mean()) / df[comb[0]].std()).values,
                                                ((df[comb[1]] - df[comb[1]].mean()) / df[comb[1]].std()).values)])



    mutuals = pd.DataFrame(mutuals)
    mutuals.columns = ['col1','col2','mutual']
    criterio = mutuals['mutual'].mean() + 2 * mutuals['mutual'].std()
    mutuals_styler = mutuals.style.applymap(color_with_criterio,subset=['mutual'])

    return (pearson_N, pearson_Y), (spearman_N, spearman_Y),(mutuals,mutuals_styler)


# function used in categoricals_comparisons
def categorical_chi_square(df, categorical, binary, observed='observed'):
    # The chi square test tests the null hypothesis that the categorical data has
    # the given frequencies
    t_shares = (df.groupby(binary)[observed].sum() / df[observed].sum()).to_dict()
    cat_shares = (df.groupby(categorical)[observed].sum()).to_dict()
    expected = []
    for row in df.itertuples():
        expected.append(cat_shares[row[1]] * float(t_shares[row[2]]))
    df['expected'] = pd.Series(expected)
    df = df[(df['observed'] > 5) & (df['expected'] > 5)]
    if df.shape[0] > 0:
        print display.display(df.head(5))
        dfree = (len(cat_shares) - 1) * (len(t_shares) - 1)
        stat, pval = scipy.stats.chisquare(df['observed'], df['expected'], ddof=dfree)

        return pval
    else:
        return 'N/A'


# Categorials / binaries features comparison, chi-square criteria
# Chi-square (non-parametric crit.) tests if observed values are close to expected values
#
# Input format:
# list of dicts, where each dict discribes 'task' for aggregator function (transforms dataframe to aggregated view according
# to group-parameter(s), parameter to aggregate and aggregation function)
#
# Output format:
#
# dataframe of chi-square pvalue for double groupping of data by categorial and binary features
#

def categoricals_comparisons(tasks):
    table = []
    for task in tasks:
        df = aggregator(task)
        categorical, binary = task['groups'][0][0], task['groups'][0][1]
        table.append([categorical, binary,
                      categorical_chi_square(df, categorical, binary, observed='observed')])
    table = pd.DataFrame(table)
    table.columns = ['categoricalF', 'binaryF', 'chi_pval']
    table = table.style.applymap(color_sign_red, subset=['chi_pval'])
    return table


# Metric multi-groups comparison (parametric/non-parametric tests)
# Function estimates multi-groups average-measures if they are the same. For metric variables of normal distribution
# One-Way ANOVA test is performed, for non-normally distributed metric variables performed non-parametric analogy of ANOVA test
# - Kruskal Wallis test
#
# Input format:
# df - dataframe
# metrics - list of metric-variables
# categories - list of categorial variables
#
# Output format:
#
# tuple of dataframed results and its styler

def disp_analysis(df, metrics, categories):
    results = []
    for metric in metrics:
        statistic, pval = stats.normaltest(df[metric].values)
        if pval > 0.05:
            for cat in categories:
                samples = [i for i in [df[df[cat] == category].drop_duplicates() \
                                           [metric].values for category in df[cat].unique()] if len(i) > 0]
                statistic, pvalue = stats.f_oneway(*samples)
                results.append([metric, cat, statistic, pvalue])
        else:
            for cat in categories:
                samples = [i for i in [df[df[cat] == category].drop_duplicates() \
                                           [metric].values for category in df[cat].unique()] if len(i) > 0]

                statistic, pvalue = stats.kruskal(*samples)
                results.append([metric, cat, statistic, pvalue])
    results = pd.DataFrame(results)
    results.columns = ['Metric_F', 'Category_F', 'statistics', 'pvalue']
    results_styler = results.style.applymap(color_sign_red, subset=['pvalue'])
    return results, results_styler


# Function checks outliers
#

def check_outs(df, col, label):
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    lower = Q1 - 1.5 * (Q3 - Q1)
    upper = Q3 + 1.5 * (Q3 - Q1)
    print 'lower border = {}, upper border = {}'.format(lower, upper)
    inners = set(df[(df[col] <= upper) & (df[col] >= lower)][label].unique())
    print 'amount of saved columns = {}'.format(float(len(inners)) / df.shape[0])
    bad_lables = set(df[label].unique()) - inners
    return bad_lables


def normalize_by_row(true_df, columns):
    df = (true_df[columns]).T
    for col in df.columns:
        tmp = df[col].max() - df[col].min()
        if tmp == 0.:
            tmp = 1.
        df[col] = (df[col] - df[col].min()) / tmp
    return df.T


def normalize_by_columns(df, columns):
    for col in columns:
        z = (df[col].max() - df[col].min())
        if z == 0.:
            z = 1.
        df[col] = (df[col] - df[col].min()) / z
    return df


def descript_cols_on_nulls(df, cols):
    categorial_cols_stats = []
    for col in cols:
        categorial_cols_stats.append([col, float(df[df[col] > 0.][col].count()) \
                                      / df.shape[0]])
    categorial_cols_stats = pd.DataFrame(categorial_cols_stats)
    categorial_cols_stats.columns = ['cat_col_name', '%notnull']
    return categorial_cols_stats


def outliers_by_outlier_cnt(subdf, all_columns):
    st = dt.now()
    outlier_pretendents = []
    for col in all_columns:
        pretendets = check_outs(subdf, col, 'user_id')

        outlier_pretendents += list(pretendets)
    outlier_pretendents = {item: outlier_pretendents.count(item) for \
                           item in set(outlier_pretendents)}
    outlier_pretendents_df = pd.DataFrame(outlier_pretendents.items())
    outlier_pretendents_df.columns = ['user_id', 'outling_facts_cnt']
    too_often_outliers = check_outs(outlier_pretendents_df, 'outling_facts_cnt', 'user_id')
    print 'percentage of outliers = {}'.format(float(len(too_often_outliers)) / \
                                               outlier_pretendents_df.shape[0])
    print 'performance = {}'.format(dt.now() - st)
    return outlier_pretendents_df, too_often_outliers


def check_if_normal(df, cols, alpha=0.05):
    cols_stats = []
    for col in cols:
        statistica, pval = stats.normaltest(df[col])
        if pval < alpha:
            status = 'Not Normal'
        else:
            status = 'Normal'
        cols_stats.append([col, statistica, pval, status])
    cols_stats = pd.DataFrame(cols_stats)
    cols_stats.columns = ['col_name', 'statistics', 'pvalue', 'H0_status']
    return cols_stats


def nonparametric_check_for_d_similarity(df1, df2, alpha=0.01):
    common_features = set(df1.columns) & set(df2.columns)
    features_stats = []
    for col in common_features:
        # H0=same central parameter
        delta_test, delta_pvalue = stats.mannwhitneyu(df1[col], df2[col])
        if delta_pvalue > alpha:
            delta = 'Same central parameter'
        else:
            delta = 'Different central parameter'
        # H0=equality of the scale parameters
        scale1_test, scale1_pval = stats.ansari(df1[col], df2[col])
        if scale1_pval > alpha:
            scale1 = 'Same scale AnsariTest'
        else:
            scale1 = 'Different scale AnsariTest'
        # H0=equality of the scale parameters
        scale2_test, scale2_pval = stats.mood(df1[col], df2[col])
        if scale2_pval > alpha:
            scale2 = 'Same scale MoodTest'
        else:
            scale2 = 'Different scale MoodTest'
        features_stats.append([col, delta_pvalue, delta, scale1_pval, scale1, scale2_pval, scale2])
    features_stats = pd.DataFrame(features_stats)
    features_stats.columns = ['col_name', 'delta_pval', 'delta_status', \
                              'scale1_pval', 'scale1_status', 'scale2_pval', 'scale2_status']
    return features_stats


def upd_dicted_dict(d, key, value):
    # presupposes value is a dict
    if key not in d:
        d[key] = {}
    d[key].update(value)
    return d


def features_corrs(df, features, pears=True, spear=True, connection='weak', alpha=0.03):
    start = dt.now()
    features_combs = [i for i in combinations(features, 2)]
    print 'features pairs to check: {}'.format(len(features_combs))
    mutual_corrs = {}

    for f_cols in tqdm(features_combs):
        ps_pval1 = None
        sp_pval1 = None
        ps_pval2 = None
        sp_pval2 = None
        if pears:
            ps_stat1, ps_pval1 = pearsonr(joined_romir[f_cols[0]], joined_romir[f_cols[1]])
            ps_stat2, ps_pval2 = pearsonr(joined_romir[f_cols[1]], joined_romir[f_cols[0]])
        if spear:
            sp_stat1, sp_pval1 = spearmanr(joined_romir[f_cols[0]], joined_romir[f_cols[1]])
            sp_stat2, sp_pval2 = spearmanr(joined_romir[f_cols[1]], joined_romir[f_cols[0]])

        if ((ps_pval1 > alpha) or (sp_pval1 > alpha)):
            mutual_corrs = upd_dicted_dict(mutual_corrs, f_cols[0], {f_cols[1]: 1})
        if ((ps_pval2 > alpha) or (sp_pval2 > alpha)):
            mutual_corrs = upd_dicted_dict(mutual_corrs, f_cols[1], {f_cols[0]: 1})

    mutual_corrs = pd.DataFrame(mutual_corrs, dtype=int)
    mutual_corrs = mutual_corrs.fillna(0)
    print 'graph nodes amount = {}'.format(mutual_corrs.shape[0])
    G_sparse = csr_matrix(mutual_corrs.values)
    comps, lables = csgraph.connected_components(G_sparse, directed=False, connection=connection)
    print 'total performance duration = ', dt.now() - start
    return comps, lables, mutual_corrs
