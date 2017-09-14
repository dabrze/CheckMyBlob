import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def read_file(path):
    print "reading", path
    return pd.read_csv(path, header=0, index_col=0)

def read_files(directory, wildcard):
    path = os.path.join(directory, wildcard)
    
    dataframes = []
    print 'checking', path

    for filepath in glob.glob(path):
        dataframes.append(read_file(filepath))

    return pd.concat(dataframes, axis=1)


def get_stats(directory, wildcard):
    df = read_files(directory, wildcard)
    print directory, wildcard.replace("*", '')
    print df.head()

    mean = df.mean(axis=1).rename('mean')
    std = df.std(axis=1).rename('std')

    stats = pd.concat([mean, std], axis=1)
    print stats.head()
    return stats

sort = True
many = False

cmb_lgmb = get_stats('cmb', 'LGBM*')
cmb_random = get_stats('cmb', 'Random*')

ter_lgmb = get_stats('ter', 'LGBM*')
ter_random = get_stats('ter', 'Random*')

cor_lgmb = get_stats('cor', 'LGBM*')
cor_random = get_stats('cor', 'Random*')

if sort:
    cmb_lgmb.sort_values('mean', ascending=True, inplace=True)
    cmb_random = cmb_random.reindex(cmb_lgmb.index)

    ter_lgmb = ter_lgmb.reindex(cmb_lgmb.index)
    ter_random = ter_random.reindex(cmb_lgmb.index)

    cor_lgmb = cor_lgmb.reindex(cmb_lgmb.index)
    cor_random = cor_random.reindex(cmb_lgmb.index)

name_map = {
    'delta_density_sqrt_E1_1': '$\Delta \sqrt{PC_3}^{D}$', 
    'delta_density_sqrt_E2_1': '$\Delta \sqrt{PC_3}^{D}$', 
    'delta_density_sqrt_E3_1': '$\Delta \sqrt{PC_3}^{D}$', 
    'delta_electrons_1': '$\Delta \\rho$ $(3.3\sigma-2.8\sigma)$', 
    'delta_electrons_2': '$\Delta \\rho$ $(3.8\sigma-3.3\sigma)$', 
    'delta_mean_1': '$\Delta \mu$ $(3.3\sigma-2.8\sigma)$',
    'delta_mean_2': '$\Delta \mu$ $(3.8\sigma-3.3\sigma)$',
    'delta_shape_segments_count_2': '$\Delta$ local maxima count $(3.8\sigma-3.3\sigma)^{S}$',
    'delta_skewness_1': '$\Delta$ skewness $(3.3\sigma-2.8\sigma)$', 
    'delta_skewness_2': '$\Delta$ skewness $(3.8\sigma-3.3\sigma)$', 
    'delta_std_1': '$\Delta SD$ $(3.3\sigma-2.8\sigma)$', 
    'delta_std_2': '$\Delta SD$ $(3.8\sigma-3.3\sigma)$', 
    'delta_volume_1': '$\Delta V$ $(3.3\sigma-2.8\sigma)$', 
    'delta_volume_2': '$\Delta V$ $(3.8\sigma-3.3\sigma)$', 
    'electrons_over_resolution_00': '$\\rho$/resolution',  
    'electrons_over_volume_00': '$\\rho$/$V$',
    'local_cut_by_mainchain_volume': 'mainchain-blob overlap',
    'local_electrons': '$\\rho^{BB}$',
    'local_mean': '$\mu^{BB}$', 
    'local_near_cut_count_C': '# of adjacent C atoms', 
    'local_near_cut_count_N': '# of adjacent N atoms', 
    'local_near_cut_count_O': '# of adjacent O atoms',
    'local_near_cut_count_S': '# of adjacent S atoms',
    'part_00_density_CI': '$CI^{D}$', 
    'part_00_density_E2_E1': '$PC_{2}$/$PC_{1}^{D}$', 
    'part_00_density_E3_E1': '$PC_{3}$/$PC_{1}^{D}$', 
    'part_00_density_E3_E2': '$PC_{3}$/$PC_{2}^{D}$', 
    'part_00_density_I2_norm': '$I_{2}^{D}$',  
    'part_00_density_I4_norm': '$I_{4}^{D}$', 
    'part_00_density_I5_norm': '$I_{5}^{D}$',  
    'part_00_density_I6': '$I_{6}^{D}$', 
    'part_00_density_O4_norm': '$O_{4}^{D}$', 
    'part_00_density_sqrt_E1': '$\sqrt{PC_1}^{D}$', 
    'part_00_density_sqrt_E2': '$\sqrt{PC_2}^{D}$',
    'part_00_density_sqrt_E3': '$\sqrt{PC_3}^{D}$', 
    'part_00_density_Z_3_0': '$Z_{3,0}^{D}$',  
    'part_00_density_Z_5_0': '$Z_{5,0}^{D}$', 
    'part_00_density_Z_6_0': '$Z_{6,0}^{D}$',  
    'part_00_density_Z_7_0': '$Z_{7,0}^{D}$',  
    'part_00_electrons': '$\\rho$', 
    'part_00_mean': '$\mu$',
    'part_00_shape_CI': '$CI_{2}^{S}$',  
    'part_00_shape_E2_E1': '$PC_{2}$/$PC_{1}^{S}$',
    'part_00_shape_FL': '$FL^{S}$',
    'part_00_shape_O5_norm': '$O_{5}^{S}$',  
    'part_00_shape_sqrt_E1': '$\sqrt{PC_1}^{S}$', 
    'part_00_shape_sqrt_E3': '$\sqrt{PC_3}^{S}$', 
    'part_00_shape_Z_1_0': '$Z_{1,0}^{S}$',
    'part_00_shape_Z_3_1': '$Z_{3,1}^{S}$', 
    'part_00_shape_Z_4_0': '$Z_{4,0}^{S}$',  
    'part_00_shape_Z_4_2': '$Z_{4,2}^{S}$',  
    'part_00_volume': '$V$', 
    'percent_cut': '% mainchain-blob overlap', 
    'resolution': 'resolution', 
    'shape_segments_count_over_volume_00': 'local maxima count/$V$', 
    'skewness_over_volume_00': 'skewness/$V$', 
    'std_over_resolution_00': '$SD$/resolution',
    'std_over_volume_01': '$SD/V$ $(3.3\sigma$)',
    'volume_over_resolution_00': '$V$/resolution', 
}


N = cor_random.shape[0]

names = list(cmb_lgmb.index)
names = [name_map[name] for name in names]
# print names
# names = [x.replace('part_00', '').replace('_', ' ').replace('00', '').replace('norm', 'normalized').replace('shape', '$S_{map}$:').replace('density', '$D_{map}$:').replace('delta', '$\Delta$').strip() for x in cmb_lgmb.index]
# names = [x.replace('Z 4 2', '$Z_{4,2}$').replace('Z 4 0', '$Z_{4,0}$').replace('Z 3 1', '$Z_{3,1}$').replace('Z 1 0', '$Z_{1,0}$').replace('Z 7 0', '$Z_{7,0}$').replace('Z 6 0', '$Z_{6,0}$').replace('Z 5 0', '$Z_{5,0}$').replace('Z 3 0', '$Z_{3,0}$') for x in names]
# names = [x.replace('E2 E1', '$PC_{2}$/$PC_{1}$').replace('E3 E1', '$PC_{3}$/$PC_{1}$').replace('E3 E2', '$PC_{3}$/$PC_{2}$') for x in names]
# names = [x.replace('sqrt E1', '$\sqrt{PC_1}$').replace('sqrt E2', '$\sqrt{PC_2}$').replace('sqrt E3', '$\sqrt{PC_3}$') for x in names]
# names = [x.replace('std', 'SD').replace(' over ', '/').replace('volume', 'V') for x in names]
# names = [x.replace(' 2', ' (3.8$\sigma-$3.3$\sigma$)').replace(' 1', ' (3.3$\sigma-$2.8$\sigma$)') for x in names]
# names = [x.replace('local near cut count O', '# of adjacent O atoms').replace('local near cut count N', '# of adjacent N atoms').replace('local near cut count C', '# of adjacent C atoms').replace('local near cut count S', '# of adjacent S atoms') for x in names]

print sorted(names)
other = ["volume"]

ind = np.arange(0, N)    # the x locations for the groups
pos = ind
space = 0.3
width = (1-space) 

fig = plt.figure(figsize=(7,16))
ax = fig.add_subplot(111)

if many:
    p_cmb_lgmb = ax.errorbar(pos, cmb_lgmb['mean'].values, yerr=cmb_lgmb['std'].values, capsize=3)
    p_cmb_random = ax.errorbar(pos, cmb_random['mean'].values, yerr=cmb_random['std'].values, capsize=3)

    p_ter_lgmb = ax.errorbar(pos, ter_lgmb['mean'].values, yerr=ter_lgmb['std'].values, capsize=3)
    p_ter_random = ax.errorbar(pos, ter_random['mean'].values, yerr=ter_random['std'].values, capsize=3)

    p_cor_lgmb = ax.errorbar(pos, cor_lgmb['mean'].values, yerr=cor_lgmb['std'].values, capsize=3)
    p_cor_random = ax.errorbar(pos, cor_random['mean'].values, yerr=cor_random['std'].values, capsize=3)
else:
    def is_gray(name):
        if 'I_' in name or 'FL' in name or 'O_{' in name or 'Z_{' in name or 'CI' in name or name in other:
            return True
        return False

    blue_x = [x if not is_gray(name) else 0 for x, name in zip(pos, names)]
    blue_y = [y if not is_gray(name) else 0 for y, name in zip(cmb_lgmb['mean'].values, names)]
    blue_std = [std if not is_gray(name) else 0 for std, name in zip(cmb_lgmb['std'].values, names)]
    p1 = ax.barh(blue_x, blue_y, xerr=blue_std, height=width, align='center', color="#1f77b4", capsize=4)
    
    gray_x = [x if is_gray(name) else 0 for x, name in zip(pos, names)]
    gray_y = [y if is_gray(name) else 0 for y, name in zip(cmb_lgmb['mean'].values, names)]
    gray_std = [std if is_gray(name) else 0 for std, name in zip(cmb_lgmb['std'].values, names)]
    p2 = ax.barh(gray_x, gray_y, xerr=gray_std, height=width, align='center', color="#999999", capsize=4)

ax.set_ylim(-width,N-1+width)
ax.set_yticks(ind)
ax.set_yticklabels(names, rotation=0)

plt.xlabel('Importance')
plt.ylabel('Feature')
if many:
    plt.legend([p_cmb_lgmb, p_cmb_random, 
            p_ter_lgmb, p_ter_random, 
            p_cor_lgmb, p_cor_random], 
           ['CMB GBM', 'CMB RandomForest',
            'Terwilliger GBM', 'Terwilliger RandomForest',
            'Carolan GBM', 'Carolan RandomForest',
           ])
else:
    plt.legend([p1, p2], ['Features designed for CheckMyBlob', 'Features used in previous studies'])

plt.tight_layout()
if many:
    fig_name = 'importances_sort.png' if sort else 'importances.png'
else:
    fig_name = 'importances_cmb_bgm.png'

fig.savefig(fig_name, dpi=300)
