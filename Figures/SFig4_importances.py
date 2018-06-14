import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

cmb_lgmb = get_stats('../Results', 'cmb_LGBMClassifier_preprocessor_feature_importance*')
cmb_random = get_stats('../Results', 'cmb_RandomForestClassifier_preprocessor_feature_importance*')

ter_lgmb = get_stats('../Results', 'tamc_LGBMClassifier_preprocessor_feature_importance*')
ter_random = get_stats('../Results', 'tamc_RandomForestClassifier_preprocessor_feature_importance*')

cor_lgmb = get_stats('../Results', 'cl_LGBMClassifier_preprocessor_feature_importance*')
cor_random = get_stats('../Results', 'cl_RandomForestClassifier_preprocessor_feature_importance*')

if sort:
    cmb_lgmb.sort_values('mean', ascending=True, inplace=True)
    cmb_random = cmb_random.reindex(cmb_lgmb.index)

    ter_lgmb = ter_lgmb.reindex(cmb_lgmb.index)
    ter_random = ter_random.reindex(cmb_lgmb.index)

    cor_lgmb = cor_lgmb.reindex(cmb_lgmb.index)
    cor_random = cor_random.reindex(cmb_lgmb.index)

name_map = {
    'delta_density_sqrt_E1_1': '$\Delta {\sqrt{\\mathrm{3^{rd}\ PCA\ eigenvalue}}}^{D}$', 
    'delta_density_sqrt_E2_1': '$\Delta {\sqrt{\\mathrm{3^{rd}\ PCA\ eigenvalue}}}^{D}$', 
    'delta_density_sqrt_E3_1': '$\Delta {\sqrt{\\mathrm{3^{rd}\ PCA\ eigenvalue}}}^{D}$', 
    'delta_electrons_1': '$\Delta$ electrons $(3.3\sigma-2.8\sigma)$',
    'delta_electrons_2': '$\Delta$ electrons $(3.8\sigma-3.3\sigma)$',
    'delta_mean_1': '$\Delta$ mean density $(3.3\sigma-2.8\sigma)$',
    'delta_mean_2': '$\Delta$ mean density $(3.8\sigma-3.3\sigma)$',
    'delta_shape_segments_count_2': '$\Delta$ local maximas ${(3.8\sigma-3.3\sigma)}^{S}$',
    'delta_skewness_1': '$\Delta$ skewness $(3.3\sigma-2.8\sigma)$', 
    'delta_skewness_2': '$\Delta$ skewness $(3.8\sigma-3.3\sigma)$', 
    'delta_std_1': '$\Delta$ standard deviation $(3.3\sigma-2.8\sigma)$',
    'delta_std_2': '$\Delta$ standard deviation $(3.8\sigma-3.3\sigma)$',
    'delta_volume_1': '$\Delta$ blob volume $(3.3\sigma-2.8\sigma)$',
    'delta_volume_2': '$\Delta$ blob volume $(3.8\sigma-3.3\sigma)$',
    'electrons_over_resolution_00': 'electrons/resolution',
    'electrons_over_volume_00': 'electrons/blob volume',
    'local_cut_by_mainchain_volume': 'mainchain-blob overlap',
    'local_electrons': 'electrons in bounding box',
    'local_mean': 'mean density in bounding box',
    'local_near_cut_count_C': '# of adjacent C atoms', 
    'local_near_cut_count_N': '# of adjacent N atoms', 
    'local_near_cut_count_O': '# of adjacent O atoms',
    'local_near_cut_count_S': '# of adjacent S atoms',
    'part_00_density_CI': 'Chiral invariant$^{D}$',
    'part_00_density_E2_E1': '${\\mathrm{2^{nd}\ PCA\ eigenvalue}/\\mathrm{1^{st}\ PCA\ eigenvalue}}^{D}$', 
    'part_00_density_E3_E1': '${\\mathrm{3^{rd}\ PCA\ eigenvalue}/\\mathrm{1^{st}\ PCA\ eigenvalue}}^{D}$', 
    'part_00_density_E3_E2': '${\\mathrm{3^{rd}\ PCA\ eigenvalue}/\\mathrm{2^{nd}\ PCA\ eigenvalue}}^{D}$', 
    'part_00_density_I2_norm': 'Moment invariant ${I_{2}}^{D}$',
    'part_00_density_I4_norm': 'Moment invariant ${I_{4}}^{D}$',
    'part_00_density_I5_norm': 'Moment invariant ${I_{5}}^{D}$',
    'part_00_density_I6': 'Moment invariant ${I_{6}}^{D}$',
    'part_00_density_O3_norm': 'Moment invariant ${O_{3}}^{D}$',
    'part_00_density_O4_norm': 'Moment invariant ${O_{4}}^{D}$',
    'part_00_density_sqrt_E1': '${\sqrt{\\mathrm{1^{st}\ PCA\ eigenvalue}}}^{D}$',
    'part_00_density_sqrt_E2': '${\sqrt{\\mathrm{2^{nd}\ PCA\ eigenvalue}}}^{D}$',
    'part_00_density_sqrt_E3': '${\sqrt{\\mathrm{3^{rd}\ PCA\ eigenvalue}}}^{D}$',
    'part_00_density_Z_3_0': 'Zernike coefficient(3,0)$^{D}$',
    'part_00_density_Z_5_0': 'Zernike coefficient(5,0)$^{D}$',
    'part_00_density_Z_6_0': 'Zernike coefficient(6,0)$^{D}$',
    'part_00_density_Z_7_0': 'Zernike coefficient(7,0)$^{D}$',
    'part_00_electrons': 'electrons',
    'part_00_mean': 'mean density',
    'part_00_shape_CI': 'Chiral invariant$^{S}$',
    'part_00_shape_E2_E1': '${\\mathrm{2^{nd}\ PCA\ eigenvalue}/\\mathrm{1^{st}\ PCA\ eigenvalue}}^{S}$',
    'part_00_shape_FL': 'Moment invariant $FL^{S}$',
    'part_00_shape_O5_norm': 'Moment invariant ${O_{5}}^{S}$',
    'part_00_shape_sqrt_E1': '${\sqrt{\\mathrm{1^{st}\ PCA\ eigenvalue}}}^{S}$', 
    'part_00_shape_sqrt_E3': '${\sqrt{\\mathrm{3^{rd}\ PCA\ eigenvalue}}}^{S}$', 
    'part_00_shape_Z_1_0': 'Zernike coefficient(1,0)$^{S}$',
    'part_00_shape_Z_2_0': 'Zernike coefficient(2,0)$^{S}$',
    'part_00_shape_Z_2_1': 'Zernike coefficient(2,1)$^{S}$',
    'part_00_shape_Z_3_1': 'Zernike coefficient(3,1)$^{S}$',
    'part_00_shape_Z_4_0': 'Zernike coefficient(4,0)$^{S}$',
    'part_00_shape_Z_4_2': 'Zernike coefficient(4,2)$^{S}$',
    'part_00_volume': 'blob volume',
    'percent_cut': '% of mainchain-blob overlap',
    'resolution': 'resolution', 
    'shape_segments_count_over_volume_00': '# of local maximas/blob volume',
    'skewness_over_volume_00': 'skewness/blob volume',
    'std_over_resolution_00': 'standard deviation/resolution',
    'std_over_volume_01': 'standard deviation/blob volume $(3.3\sigma)$',
    'volume_over_resolution_00': 'blob volume/resolution',
}


N = cor_random.shape[0]

names = list(cmb_lgmb.index)
names = [name_map[name] for name in names]

print sorted(names)
other = ["blob volume"]

matplotlib.rcParams.update({'font.size': 4.5})
ind = np.arange(0, N)    # the x locations for the groups
pos = ind
space = 0.25
width = (1-space) 

fig = plt.figure(figsize=(3.15, 5.27))
ax = fig.add_subplot(111)


def is_gray(name):
    if 'I_' in name or 'FL' in name or 'O_{' in name or 'Zernike' in name or 'Chiral' in name or name in other:
        return True
    return False

blue_x = [x if not is_gray(name) else 0 for x, name in zip(pos, names)]
blue_y = [y if not is_gray(name) else 0 for y, name in zip(cmb_lgmb['mean'].values, names)]
blue_std = [std if not is_gray(name) else 0 for std, name in zip(cmb_lgmb['std'].values, names)]
p1 = ax.barh(blue_x, blue_y, xerr=blue_std, height=width, align='center', color="#1f77b4", capsize=1.5, error_kw={'capthick': 0.4, 'elinewidth': 0.4 })

gray_x = [x if is_gray(name) else 0 for x, name in zip(pos, names)]
gray_y = [y if is_gray(name) else 0 for y, name in zip(cmb_lgmb['mean'].values, names)]
gray_std = [std if is_gray(name) else 0 for std, name in zip(cmb_lgmb['std'].values, names)]
p2 = ax.barh(gray_x, gray_y, xerr=gray_std, height=width, align='center', color="#999999", capsize=1.5, error_kw={'capthick': 0.4, 'elinewidth': 0.4 })

ax.set_ylim(-width,N-1+width)
ax.set_yticks(ind)
ax.set_yticklabels(names, rotation=0)

plt.xlabel('Feature importance')
#plt.ylabel('Feature')

plt.legend([p1, p2], ['Designed for CheckMyBlob', 'Also used in previous studies'])

plt.tight_layout()
fig_name = 'SFig4/SFig4.png'
fig.savefig(fig_name, dpi=300)

fig_name = 'SFig4/SFig4.svg'
fig.savefig(fig_name, dpi=300)

fig_name = 'SFig4/SFig4.eps'
fig.savefig(fig_name, rasterize=False, dpi=300)

