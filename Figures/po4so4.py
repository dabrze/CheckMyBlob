import glob
import os

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from matplotlib.ticker import FixedLocator, FixedFormatter

import pandas as pd

SCALE = 1
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = SCALE * 6
plt.rcParams['axes.labelsize'] = SCALE * 6
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = SCALE * 7
plt.rcParams['xtick.labelsize'] = SCALE * 5
plt.rcParams['ytick.labelsize'] = SCALE * 5
plt.rcParams['legend.fontsize'] = SCALE * 6
plt.rcParams['figure.titlesize'] = SCALE * 7
plt.rcParams['lines.markersize'] = 2.0  # 6.0
plt.rcParams['lines.linewidth'] = 0.75
X_SIZE = SCALE * 4.75
Y_SIZE = SCALE * 2
DPI = 600


def read_file(path, separator=',', index_col=0, dtype={'is_correct': np.bool}):
    print "reading", path
    return pd.read_csv(path, header=0, index_col=index_col, sep=separator, keep_default_na=False, na_values=[''],
                       dtype=dtype)


def read_files(directory, wildcard):
    path = os.path.join(directory, wildcard)

    dataframes = []
    print 'checking', path

    for filepath in glob.glob(path):
        dataframes.append(read_file(filepath))

    return pd.concat(dataframes, axis=0)


def get_stats(directory, wildcard, group_by_column='resolution', low=1.0, high=4.0, step=0.1, digits=1,
              ligands='ligands.csv', bins=None, value_column="is_correct"):
    df = read_files(directory, wildcard)
    print directory, wildcard.replace("*", '')
    print df.head()

    print group_by_column, 'all', df.loc[:, group_by_column].count(), 'empty', df[group_by_column].isnull().sum()
    df = df.ix[~df[group_by_column].isnull()]
    print group_by_column, 'all', df.loc[:, group_by_column].count(), 'empty', df[group_by_column].isnull().sum()

    # keep only SO4 and PO4
    df = df[((df['y_pred'] == 'SO4') | (df['y_pred'] == 'PO4')) & ((df['y_true'] == 'SO4') | (df['y_true'] == 'PO4'))]
    df["is_SO4"] = (df['y_true'] == 'SO4')
    df["y_pred_prob"] = 1.0-df['y_pred_prob']
    print df.head()

    if bins is None:
        df.loc[:, group_by_column] = ((df.loc[:, group_by_column] / step).round(digits) * step)
    else:
        # round needed to resolve floating point numerical issues
        df.loc[:, group_by_column] = df.loc[:, group_by_column].round(5)
        for lo, up in zip(bins[0:-1], bins[1:]):
            print lo, up
            df.loc[(df.loc[:, group_by_column] > np.round(lo,5)) & (df.loc[:, group_by_column] <= np.round(up,5)), group_by_column] = 0.5 * (
                    lo + up)
    print df.head()

    if bins is None:
        low_mean = df[df[group_by_column] < low].loc[:,value_column].mean(skipna=True)
        high_mean = df[df[group_by_column] > high].loc[:,value_column].mean(skipna=True)

        low_std = df[df[group_by_column] < low].loc[:,value_column].std(skipna=True)
        high_std = df[df[group_by_column] > high].loc[:,value_column].std(skipna=True)

        low_count = df[df[group_by_column] < low].loc[:,value_column].count()
        high_count = df[df[group_by_column] > high].loc[:,value_column].count()
    else:
        low_mean = df[df[group_by_column] <= bins[1]].loc[:,value_column].mean(skipna=True)
        high_mean = df[df[group_by_column] > bins[-2]].loc[:,value_column].mean(skipna=True)

        low_std = df[df[group_by_column] <= bins[1]].loc[:,value_column].std(skipna=True)
        high_std = df[df[group_by_column] > bins[-2]].loc[:,value_column].std(skipna=True)

        low_count = df[df[group_by_column] <= bins[1]].loc[:,value_column].count()
        high_count = df[df[group_by_column] > bins[-2]].loc[:,value_column].count()

    print 'edge', low_mean, high_mean, low_count, high_count

    if bins is None:
        df = df[(df[group_by_column] >= low) & (df[group_by_column] <= high)]
    else:
        df = df[(df[group_by_column] >= bins[1]) & (df[group_by_column] <= bins[-2])]

    group = df.groupby(group_by_column)

    mean = group[value_column].mean()
    std = group[value_column].std()
    count = group[value_column].count()

    print mean, high_mean

    if bins is None:
        mean.loc[low - step] = low_mean
        mean.loc[high + step] = high_mean

        std.loc[low - step] = low_std
        std.loc[high + step] = high_std

        count.loc[low - step] = low_count
        count.loc[high + step] = high_count
    else:
        cent_lo = 0.5 * (bins[0] + bins[1])
        cent_hi = 0.5 * (bins[-2] + bins[-1])
        mean.loc[cent_lo] = low_mean
        mean.loc[cent_hi] = high_mean

        std.loc[cent_lo] = low_std
        std.loc[cent_hi] = high_std

        count.loc[cent_lo] = low_count
        count.loc[cent_hi] = high_count

    mean.sort_index(inplace=True)
    std.sort_index(inplace=True)
    count.sort_index(inplace=True)

    print '##########'
    print count
    print mean
    print '##########'
    cumulative = np.cumsum(mean * count) / np.cumsum(count)

    stats = pd.concat([mean, std, count, cumulative], axis=1)
    stats.columns = ['accuracy', 'std', 'count', 'cumulative']
    stats.sort_index(inplace=True)
    print stats
    return stats


filename = {'resolution': 'FigS5_b.png', 'threshold': 'FigS5_a.png'}
low = {'resolution': 1.0, 'threshold': 0.1}
high = {'resolution': 2.8, 'threshold': 0.7}
step = {'resolution': 0.2, 'threshold': 0.05}
right_lte = {'resolution': True, 'threshold': True}
left_lte = {'resolution': True, 'threshold': True}
column_name = {'resolution': 'Resolution', 'threshold': 'Uncertainty'}
digits = {'resolution': 2, 'threshold': 2}
bins = {'resolution': np.arange(low['resolution'] - step['resolution'],
                           high['resolution'] + 3 * step['resolution'],
                           step['resolution']),
        'threshold': np.arange(low['threshold'] - step['threshold'],
                           high['threshold'] + 3 * step['threshold'],
                           step['threshold'])}
data_column_name = {'resolution': 'resolution', 'threshold': 'y_pred_prob'}

delete_x_ticks = {'resolution': False, 'threshold': False}
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for column in ['resolution', 'threshold']:
    cmb = get_stats('../Results', 'cmb_StackingCVClassifier_preprocessor_predictions_*',
        data_column_name[column],
        low[column],
        high[column],
        step[column],
        digits=0,
        bins=bins[column],
    )

    cmb_baseline = get_stats('../Results', 'cmb_StackingCVClassifier_preprocessor_predictions_*',
        data_column_name[column],
        low[column],
        high[column],
        step[column],
        digits=0,
        bins=bins[column],
        value_column='is_SO4',
    )


    fig = plt.figure(figsize=(X_SIZE, Y_SIZE))
    ax = fig.add_subplot(111)


    p_count_cmb = ax.bar(cmb.index.values - 0.2 * step[column], (cmb['accuracy'].values), width=0.35 * step[column],
                          align='center', color=colors[0] + "66")
    p_count_baseline = ax.bar(cmb_baseline.index.values + 0.2 * step[column], (cmb_baseline['accuracy'].values), width=0.35 * step[column],
                           align='center', color=colors[1] + "66")

    p_cmb = ax.errorbar(cmb.index.values, cmb['cumulative'].values, color=colors[0], marker='o')
    p_baseline = ax.errorbar(cmb_baseline.index.values, cmb_baseline['cumulative'].values, color=colors[1], marker='s')

    if bins[column] is None:
        ax.set_xlim(low[column] - 1.5 * step[column], high[column] + 1.5 * step[column])
    else:
        ax.set_xlim(low[column] - 1.0 * step[column], high[column] + 2.0 * step[column])
    ax.set_ylim(0.75, 1.0001)

    xticks = np.arange(low[column] - 1 * step[column], high[column] + 1 * step[column] + 0.0001, step[column])
    xticks_names = list(xticks)

    xticks_minor = np.arange(low[column] - 0.5 * step[column], high[column] + 1.5 * step[column] + 0.0001, step[column])

    print len(xticks), len(xticks_names), len(xticks_minor)

    if bins[column] is not None:
        xticks_names = [u"(%.1f\u2013%.1f]" % (xt, xt + step[column]) for xt in xticks]

    if left_lte[column]:
        xticks_names[0] = u"< %.1f" % (low[column])  # \u2264
        if bins[column] is not None:
            xticks_names[0] = u"\u2264 %.1f" % (low[column])  # \u2264
    else:
        xticks_names[0] = u"\u2264 %.1f" % (low[column] - step[column])  # \u2264

    if right_lte[column]:
        xticks_names[-1] = u"> %.1f" % (high[column])  # \u2265
        if bins[column] is not None:
            xticks_names[-1] = u"> %.1f" % (high[column] + step[column])  # \u2264

    if delete_x_ticks[column]:
        for i_tick, x_name in enumerate(xticks_names):
            if i_tick != 0 and i_tick != len(xticks_names) - 1 and i_tick % 2 == 1:
                xticks_names[i_tick] = ''

    ax.set_ylabel('Recognition Rate')
    ax.set_xlabel(column_name[column])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_names)
    if bins[column] is not None:
        ax.set_xticklabels('')
        print xticks_minor
        print xticks_names
        ax.xaxis.set_minor_locator(FixedLocator(xticks_minor))
        ax.xaxis.set_minor_formatter(FixedFormatter(xticks_names))
        ax.tick_params(axis='x', which='minor', length=0)

    # ax.set_yticks(np.arange(0.0, 0.801, step[column]))

    plt.legend([p_cmb, p_baseline, p_count_cmb, p_count_baseline],
               ['Cumulative stacking on $\it{CMB}$ limited to SO4, PO4',
                'Cumulative majority stub on $\it{CMB}$ limited to SO4, PO4',
                'Stacking on $\it{CMB}$ limited to SO4, PO4',
                'Majority stub on $\it{CMB}$ limited to SO4, PO4',
                ]
    )

    plt.tight_layout()
    fig.savefig(filename[column], dpi=DPI)
    fig.savefig(filename[column].replace('png', 'svg'), dpi=DPI)
    fig.savefig(filename[column].replace('png', 'eps'), rasterize=False, dpi=DPI)
