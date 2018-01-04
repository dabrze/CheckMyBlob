import glob
import os

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.ticker import FixedLocator, FixedFormatter

import pandas as pd

SCALE = 1
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = SCALE*6
plt.rcParams['axes.labelsize'] = SCALE*6
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = SCALE*7
plt.rcParams['xtick.labelsize'] = SCALE*5
plt.rcParams['ytick.labelsize'] = SCALE*5
plt.rcParams['legend.fontsize'] = SCALE*6
plt.rcParams['figure.titlesize'] = SCALE*7
plt.rcParams['lines.markersize'] = 2.0  #6.0
plt.rcParams['lines.linewidth'] = 0.75
X_SIZE = SCALE*4.75
Y_SIZE = SCALE*2
DPI = 600

def read_file(path, separator=',', index_col=0, dtype={'is_correct': np.bool}):
    print "reading", path
    return pd.read_csv(path, header=0, index_col=index_col, sep=separator, keep_default_na=False, na_values=[''], dtype=dtype)

def parse_ligand(ligands='ligands.csv'):
    atoms = []
    with open(ligands, 'r') as ligs:
        for line in ligs.read().splitlines()[1:]:
            s_line = line.split(';')
            atoms.append((s_line[0], int(s_line[3])))
    df = pd.DataFrame(atoms)
    df.columns = ['CODE', 'non_h_atoms']
    return df

def read_files(directory, wildcard):
    path = os.path.join(directory, wildcard)

    dataframes = []
    print 'checking', path

    for filepath in glob.glob(path):
        dataframes.append(read_file(filepath))

    return pd.concat(dataframes, axis=0)

def get_stats(directory, wildcard, group_by_column='resolution', low=1.0, high=4.0, step=0.1, digits=1, ligands='ligands.csv', bins=None):
    ligands = parse_ligand(ligands)
    df = read_files(directory, wildcard)
    df = df.merge(ligands, left_on="y_true", right_on="CODE", how="left")
    print directory, wildcard.replace("*", '')
    print df.head()

    print group_by_column, 'all', df.loc[:, group_by_column].count(), 'empty', df[group_by_column].isnull().sum()
    df = df.ix[~df[group_by_column].isnull()]
    print group_by_column, 'all', df.loc[:, group_by_column].count(), 'empty', df[group_by_column].isnull().sum()

    if bins is None:
        df.loc[:, group_by_column] = ((df.loc[:, group_by_column]/step).round(digits)*step)
    else:
        for lo, up in zip(bins[0:-1], bins[1:]):
            print lo, up
            df.loc[(df.loc[:,group_by_column] > lo) & (df.loc[:,group_by_column] <= up), group_by_column] = 0.5*(lo+up)
    print df.head()

    if bins is None:
        low_mean = df[df[group_by_column] < low].is_correct.mean(skipna=True)
        high_mean = df[df[group_by_column] > high].is_correct.mean(skipna=True)

        low_std = df[df[group_by_column] < low].is_correct.std(skipna=True)
        high_std = df[df[group_by_column] > high].is_correct.std(skipna=True)

        low_count = df[df[group_by_column] < low].is_correct.count()
        high_count = df[df[group_by_column] > high].is_correct.count()
    else:
        low_mean = df[df[group_by_column] < bins[1]].is_correct.mean(skipna=True)
        high_mean = df[df[group_by_column] > bins[-2]].is_correct.mean(skipna=True)

        low_std = df[df[group_by_column] < bins[1]].is_correct.std(skipna=True)
        high_std = df[df[group_by_column] > bins[-2]].is_correct.std(skipna=True)

        low_count = df[df[group_by_column] < bins[1]].is_correct.count()
        high_count = df[df[group_by_column] > bins[-2]].is_correct.count()

    print 'edge', low_mean, high_mean, low_count, high_count

    if bins is None:
        df = df[(df[group_by_column] >= low) & (df[group_by_column] <= high)]
    else:
        df = df[(df[group_by_column] >= bins[1]) & (df[group_by_column] <= bins[-2])]

    group = df.groupby(group_by_column)

    mean = group.is_correct.mean()
    std = group.is_correct.std()
    count = group.is_correct.count()

    print mean, high_mean

    if bins is None:
        mean.loc[low-step] = low_mean
        mean.loc[high+step] = high_mean

        std.loc[low-step] = low_std
        std.loc[high+step] = high_std

        count.loc[low-step] = low_count
        count.loc[high+step] = high_count
    else:
        cent_lo = 0.5*(bins[0]+bins[1])
        cent_hi = 0.5*(bins[-2]+bins[-1])
        mean.loc[cent_lo] = low_mean
        mean.loc[cent_hi] = high_mean

        std.loc[cent_lo] = low_std
        std.loc[cent_hi] = high_std

        count.loc[cent_lo] = low_count
        count.loc[cent_hi] = high_count

    print '##########'
    print count
    print mean
    print '##########'

    stats = pd.concat([mean, std, count], axis=1)
    stats.columns = ['accuracy', 'std', 'count']
    stats.sort_index(inplace=True)
    print stats
    return stats

filename = {'resolution': 'Fig4_a.png', 'rscc': 'Fig4_b.png', 'non_h_atoms': 'Fig4_c.png'}
low = {'resolution': 1.0, 'rscc':0.7, 'non_h_atoms': 5}
high = {'resolution': 4.0, 'rscc':0.9, 'non_h_atoms': 50}
step ={'resolution': 0.1, 'rscc':0.1, 'non_h_atoms': 5}
right_lte = {'resolution': True, 'rscc': False, 'non_h_atoms': True}
left_lte = {'resolution': True, 'rscc': False, 'non_h_atoms': True}
column_name = {'resolution': 'Resolution', 'rscc': 'RSCC', 'non_h_atoms': 'Non H atoms'}
digits = {'resolution': 2, 'rscc': 2, 'non_h_atoms': 1}
bins = {'resolution': None, 'rscc': None,
        'non_h_atoms': range(low['non_h_atoms']-step['non_h_atoms'], high['non_h_atoms']+3*step['non_h_atoms'],step['non_h_atoms'])}

delete_x_ticks = {'resolution': True, 'rscc': False, 'non_h_atoms': False}

for column in ['resolution', 'rscc', 'non_h_atoms']:
    cmb = get_stats('../Results', 'cmb_StackingCVClassifier_preprocessor_predictions_*', column, low[column], high[column], step[column], digits=0, bins=bins[column])
    tamc = get_stats('../Results', 'tamc_StackingCVClassifier_preprocessor_predictions_*', column, low[column], high[column], step[column], digits=0, bins=bins[column])
    cl = get_stats('../Results', 'cl_StackingCVClassifier_preprocessor_predictions_*', column, low[column], high[column], step[column], digits=0, bins=bins[column])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig = plt.figure(figsize=(X_SIZE, Y_SIZE))
    ax = fig.add_subplot(111)

    ax2 = ax.twinx()
    if bins[column] is None:
        ax2.set_xlim(low[column]-1.5*step[column], high[column]+1.5*step[column])
    else:
        ax2.set_xlim(low[column]-1.0*step[column], high[column]+2.0*step[column])

    p_count_cmb = ax2.bar(cmb.index.values-0.3*step[column], (cmb['count'].values), width=0.25*step[column], align='center', color=colors[0]+"66")
    p_count_tamc = ax2.bar(tamc.index.values+0.0*step[column], (tamc['count'].values), width=0.25*step[column], align='center', color=colors[1]+"66")
    p_count_cl = ax2.bar(cl.index.values+0.3*step[column], (cl['count'].values), width=0.25*step[column], align='center', color=colors[2]+"66")
    ax2.set_ylabel('Number of examples')

    p_cmb = ax.errorbar(cmb.index.values, cmb['accuracy'].values, color=colors[0], marker='o')
    p_tamc = ax.errorbar(tamc.index.values, tamc['accuracy'].values, color=colors[1], marker='s')
    p_cl = ax.errorbar(cl.index.values, cl['accuracy'].values, color=colors[2], marker='^')

    if bins[column] is None:
        ax.set_xlim(low[column]-1.5*step[column], high[column]+1.5*step[column])
    else:
        ax.set_xlim(low[column]-1.0*step[column], high[column]+2.0*step[column])
    ax.set_ylim(0, 1)

    xticks= np.arange(low[column]-1*step[column], high[column]+1*step[column]+0.0001, step[column])
    xticks_names = list(xticks)

    xticks_minor = np.arange(low[column]-0.5*step[column], high[column]+1.5*step[column]+0.0001, step[column])

    print len(xticks), len(xticks_names), len(xticks_minor)

    if bins[column] is not None:
        xticks_names = [u"(%.0f\u2013%.0f]" % (xt, xt+step[column]) for xt in xticks]

    if left_lte[column]:
        xticks_names[0] = u"< %.1f" % (low[column]) #\u2264
        if bins[column] is not None:
            xticks_names[0] = u"\u2264 %.0f" % (low[column]) #\u2264
    else:
        xticks_names[0] = u"\u2264 %.1f" % (low[column]-step[column]) #\u2264

    if right_lte[column]:
        xticks_names[-1] = u"> %.1f" % (high[column]) #\u2265
        if bins[column] is not None:
            xticks_names[-1] = u"> %.0f" % (high[column]+step[column]) #\u2264

    if delete_x_ticks[column]:
       for i_tick, x_name in enumerate(xticks_names):
           if i_tick != 0 and i_tick != len(xticks_names)-1 and i_tick % 2 == 1:
               xticks_names[i_tick] = ''

    ax.set_ylabel('Accuracy')
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

    #ax.set_yticks(np.arange(0.0, 0.801, step[column]))

    plt.legend([p_cmb, p_tamc, p_cl], ['Stacking on $\it{CMB}$ dataset', 'Stacking on $\it{TAMC}$ dataset', 'Stacking on $\it{CL}$ dataset'])

    plt.tight_layout()
    fig.savefig(filename[column], dpi=DPI)
    fig.savefig(filename[column].replace('png', 'svg'), dpi=DPI)
    fig.savefig(filename[column].replace('png', 'eps'), rasterize=False, dpi=DPI)
