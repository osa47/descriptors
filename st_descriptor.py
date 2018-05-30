#!/usr/bin/env
import pandas as pd
import numpy as np
from scipy import stats
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

    return groups_stats,styler


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
# two tuples
# - first tuple of 3 tables
# - second tuple contains stylers for each table
#

def correlations(df, features):
    # Pearson correlation measures the linear association between continuous variables
    # Spearman's correlation measures monotonic association (only strictly increasing or decreasing,
    # but not mixed) between two variables and relies on the rank order of values.

    # The p-value roughly indicates the probability of an uncorrelated system producing datasets

    def color_with_criterio(val):
        color = 'red' if val >= criterio else 'black'
        return 'color: %s' % color

    mutuals = np.zeros((len(features), len(features)))
    pearson = np.zeros((len(features), len(features)))
    spearman = np.zeros((len(features), len(features)))
    for n1, feat1 in enumerate(features):
        pearson_flag = False
        _, pval = stats.normaltest(df[feat1].values)
        if pval > 0.05:
            pearson_flag = True
        spearman[n1][n1] = 0.
        for n2, feat2 in enumerate(features):
            if n2 != n1:
                if pearson_flag:
                    _, pval = stats.normaltest(df[feat2].values)
                    if pval > 0.05:
                        pear, pearval = stats.pearsonr(df[feat1].values, df[feat2].values)
                        pearson[n1][n2] = pearval
                spear, spearval = stats.spearmanr(df[feat1].values, df[feat2].values)
                spearman[n1][n2] = spearval
            mutuals[n1][n2] = metrics.mutual_info_score(((df[feat1] - df[feat1].mean()) / df[feat1].std()).values,
                                                        ((df[feat2] - df[feat2].mean()) / df[feat2].std()).values)
    pearson = pd.DataFrame(pearson)
    pearson.index = features
    pearson.columns = features
    pearson_styler = pearson.style.applymap(color_sign_red)

    spearman = pd.DataFrame(spearman)
    spearman.index = features
    spearman.columns = features
    spearman_styler = spearman.style.applymap(color_sign_red)

    criterio = mutuals.mean() + 2 * mutuals.std()
    mutuals = pd.DataFrame(mutuals)
    mutuals.index = features
    mutuals.columns = features
    mutuals_styler = mutuals.style.applymap(color_with_criterio)

    return (pearson, spearman, mutuals),(pearson_styler,spearman_styler,mutuals_styler)


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
    results = results.style.applymap(color_sign_red, subset=['pvalue'])
    return results