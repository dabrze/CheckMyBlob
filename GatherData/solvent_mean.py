import os
import numpy as np
import pandas as pd
import glob

# draw
try:
    MATPLOTLIB = True
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as err:
    print "MATPLOT LIB IMPOR ERR"
    print err
    MATPLOTLIB = False

from scipy import stats


STD_GRAPHS = False and MATPLOTLIB  # True
CLUSTERING_GRAPHS = False and MATPLOTLIB  # True
PARTITIONING_GRAPHS = False and MATPLOTLIB
CORRELATION_SHAPE = False and MATPLOTLIB


def print_graph(x_data, y_data, x_key, y_key, result_data, fig_filename, colors=None):
    fo_col = result_data['fo_col'][0]
    fc_col = result_data['fc_col'][0]
    grid_space = result_data['grid_space'][0]
    solvent_radius = result_data['solvent_radius'][0]
    solvent_opening_radius = result_data['solvent_opening_radius'][0]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    #fit = np.polyfit(x_data, y_data, 1)
    fit_fn = np.poly1d([slope, intercept])
    if colors is None:
        plt.scatter(x_data, y_data, label=y_key)
    else:
        plt.scatter(x_data, y_data, label=y_key, c=colors, cmap=plt.cm.spectral)
    plt.plot(x_data, fit_fn(x_data), '-k', label='fit, a=%.3f, b=%.3f, r^2=%.3f' % (slope, intercept, r_value**2))
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title('MAP %s/%s solvent radius %.1f, opening %.1f, grid %.1f' % (fo_col, fc_col, solvent_radius, solvent_opening_radius, grid_space))
    plt.legend()
    plt.gcf().set_size_inches(15, 10)
    plt.savefig(fig_filename, dpi=150)
    plt.clf()


def print_partitioning(x_data, y_data, labels):
    plot_x = np.array(x_data)
    plot_y = np.array(y_data)
    plt.plot(plot_x, plot_y, '-k')
    for x, y, lab in zip(plot_x, plot_y, labels):
        plt.annotate(
            '%d' % (lab),
            xy=(x, y),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center',
            va='top')


def get_data_keys(filename):
    result_file = open(filename, 'r')
    result_data_keys = {}

    line = result_file.read().splitlines()[0]
    line_split = line.split(";")

    for i_key, key in enumerate(line_split):
        result_data_keys[i_key] = key

    result_file.close()

    return result_data_keys


def get_data(filename, result_data_keys):
    result_file = open(filename, 'r')
    result_data = {}

    for i_key, key in result_data_keys.iteritems():
        result_data[key] = []

    for line in result_file.read().splitlines()[1:]:
        line = line.strip()
        if len(line) > 0:
            line_split = line.split(";")
            for i_col, value in enumerate(line_split):
                conv_value = 0.0
                try:
                    #if value == 'nan':
                    #    conv_value = 0.0
                    #else:
                    conv_value = float(value)
                    #if conv_value > 10e36:
                    #    conv_value = 10e36
                    #elif conv_value < -10e36:
                    #    conv_value = -10e36
                except:
                    conv_value = value
                result_data[result_data_keys[i_col]].append(conv_value)
    result_file.close()
    return result_data


def prepare_graphs(data_dir, output_dir):

    file_dir = data_dir

    if STD_GRAPHS is True:
        global_data_dir = os.path.join(file_dir, 'global_data.txt')

        # run
        result_data_keys = get_data_keys(global_data_dir)
        result_data = get_data(global_data_dir, result_data_keys)

        x_data = np.array(result_data['FoFc_std'])
        y_data = 3*np.array(result_data['TwoFoFc_bulk_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_FoFc_std.png')
        print_graph(x_data, y_data, 'FoFc_std', '3*2FoFc_bulk_std', result_data, png)

        x_data = np.array(result_data['TwoFoFc_std'])
        y_data = 3*np.array(result_data['TwoFoFc_bulk_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_2FoFc_std.png')
        print_graph(x_data, y_data, '2FoFc_std', '3*2FoFc_bulk_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['TwoFoFc_bulk_std'])/np.array(result_data['TwoFoFc_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_d_2FoFc_resolution.png')
        print_graph(x_data, y_data, 'resolution', '2FoFc_bulk_std/2FoFc_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['TwoFoFc_bulk_std'])/np.array(result_data['FoFc_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_d_FoFc_resolution.png')
        print_graph(x_data, y_data, 'resolution', '2FoFc_bulk_std/FoFc_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = 3*np.array(result_data['TwoFoFc_bulk_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_resolution.png')
        print_graph(x_data, y_data, 'resolution', '3*2FoFc_bulk_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = 3*np.array(result_data['FoFc_std'])
        png = os.path.join(output_dir, 'STD_FoFc_std_resolution.png')
        print_graph(x_data, y_data, 'resolution', '3*FoFc_std', result_data, png)

        # Fo map
        x_data = np.array(result_data['FoFc_std'])
        y_data = 3*np.array(result_data['Fo_bulk_std'])
        png = os.path.join(output_dir, 'STD_Fo_bulk_FoFc_std.png')
        print_graph(x_data, y_data, 'FoFc_std', '3*Fo_bulk_std', result_data, png)

        x_data = np.array(result_data['Fo_std'])
        y_data = 3*np.array(result_data['Fo_bulk_std'])
        png = os.path.join(output_dir, 'STD_Fo_bulk_Fo_std.png')
        print_graph(x_data, y_data, 'Fo_std', '3*Fo_bulk_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['Fo_bulk_std'])/np.array(result_data['Fo_std'])
        png = os.path.join(output_dir, 'STD_Fo_bulk_d_Fo_resolution.png')
        print_graph(x_data, y_data, 'resolution', 'Fo_bulk_std/Fo_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['Fo_bulk_std'])/np.array(result_data['FoFc_std'])
        png = os.path.join(output_dir, 'STD_Fo_bulk_d_FoFc_resolution.png')
        print_graph(x_data, y_data, 'resolution', 'Fo_bulk_std/FoFc_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = 3*np.array(result_data['Fo_bulk_std'])
        png = os.path.join(output_dir, 'STD_Fo_bulk_resolution.png')
        print_graph(x_data, y_data, 'resolution', '3*Fo_bulk_std', result_data, png)

        x_data = np.array(result_data['solvent_ratio'])
        y_data = 3*np.array(result_data['TwoFoFc_bulk_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_bulk_std_solvent_ratio.png')
        print_graph(x_data, y_data, 'solvent_ratio', '3*2FoFc_bulk_std', result_data, png)

        x_data = np.array(result_data['solvent_ratio'])
        y_data = 3*np.array(result_data['TwoFoFc_std'])
        png = os.path.join(output_dir, 'STD_2FoFc_std_solvent_ratio.png')
        print_graph(x_data, y_data, 'solvent_ratio', '3*2FoFc_std', result_data, png)

        x_data = np.array(result_data['solvent_ratio'])
        y_data = np.array(result_data['FoFc_std'])
        png = os.path.join(output_dir, 'STD_FoFc_std_solvent_ratio.png')
        print_graph(x_data, y_data, 'solvent_ratio', 'FoFc_std', result_data, png)

        x_data = np.array(result_data['solvent_ratio'])
        y_data = np.array(result_data['TwoFoFc_bulk_std'])/np.array(result_data['FoFc_std'])
        png = os.path.join(output_dir, 'STD_FoFc_std_d_FoFc_solvent_ratio.png')
        print_graph(x_data, y_data, 'solvent_ratio', '2FoFc_bulk_std/FoFc_std', result_data, png)

        x_data = np.array(result_data['solvent_ratio'])
        y_data = np.array(result_data['TwoFoFc_bulk_std'])/np.array(result_data['TwoFoFc_std'])
        png = os.path.join(output_dir, 'STD_FoFc_std_d_2FoFc_solvent_ratio.png')
        print_graph(x_data, y_data, 'solvent_ratio', '2FoFc_bulk_std/2FoFc_std', result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['solvent_ratio'])
        png = os.path.join(output_dir, 'STD_resolution_solvent_ratio.png')
        print_graph(x_data, y_data, 'resolution', 'solvent_ratio', result_data, png)

    # ===========================
    if CLUSTERING_GRAPHS is True:
        MANY_EEAMPLES_THRESHOLD = 10
        global_data_dir = os.path.join(file_dir, 'all_summary.txt')

        # run
        result_data_keys = get_data_keys(global_data_dir)
        result_data = get_data(global_data_dir, result_data_keys)

        for key, val in result_data_keys.iteritems():
            print key, val

        x_data = np.array(result_data['local_volume'])
        y_data = np.array(result_data['local_std'])/np.array(result_data['local_mean'])
        colors_set = set(result_data['res_name'])
        colors_dict = {}
        for ii, val in enumerate(colors_set):
            colors_dict[val] = ii
        colors = []
        for res in result_data['res_name']:
            colors.append(colors_dict[res])
        colors = np.array(colors)

        count = np.bincount(np.array(colors))
        ii = np.nonzero(count)[0]

        res_count = {}
        mask = np.zeros_like(colors, dtype=bool)
        for jj, label in enumerate(colors):
            res_count[result_data['res_name'][jj]] = count[label]
            if count[label] > MANY_EEAMPLES_THRESHOLD:
                mask[jj] = True

        for key in sorted(res_count.keys()):
            if res_count[key] > MANY_EEAMPLES_THRESHOLD:
                print key, res_count[key]

        print x_data.shape
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        print x_data_mask.shape

        plt.xlim(0.0, 150)
        plt.ylim(0.1, 1)
        diff_png = os.path.join(output_dir, 'CLUSTER_local_volume_mean_std.png')
        print_graph(x_data_mask, y_data_mask, 'local_volume', 'local_std/local_mean', result_data, diff_png, colors=colors_mask)

        for key in sorted(res_count.keys()):
            if res_count[key] > MANY_EEAMPLES_THRESHOLD:
                plt.xlim(0.0, 150)
                plt.ylim(0.1, 1)
                res_key = colors_dict[key]
                res_mask = colors == res_key
                x_data_mask = x_data[res_mask]
                y_data_mask = y_data[res_mask]
                colors_mask = colors[res_mask]
                png = os.path.join(output_dir, '%s_local_volume_mean_std.png' % key)
                print_graph(x_data_mask, y_data_mask, 'local_volume', '%s local_std/local_mean' % key, result_data, png)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['local_parts'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_resol_parts.png')
        print_graph(x_data_mask, y_data_mask, 'resolution', 'local_parts', result_data, png, colors=colors_mask)

        x_data = np.array(result_data['resolution'])
        y_data = np.array(result_data['local_volume'])
        png = os.path.join(output_dir, 'local_resol_vol.png')
        print_graph(x_data, y_data, 'resolution', 'local_volume', result_data, png, colors=colors)


        x_data = np.array(result_data['local_electrons'])
        y_data = np.array(result_data['local_std'])/np.array(result_data['local_mean'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_electrons_std_mean.png')
        print_graph(x_data_mask, y_data_mask, 'local_electrons', 'local_std/local_mean', result_data, png, colors=colors_mask)


        x_data = np.array(result_data['local_max'])
        y_data = np.array(result_data['local_std'])/np.array(result_data['local_mean'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_max_std_mean.png')
        print_graph(x_data_mask, y_data_mask, 'local_max', 'local_std/local_mean', result_data, png, colors=colors_mask)

        x_data = np.array(result_data['local_volume'])
        y_data = np.array(result_data['local_max'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_volume_max.png')
        print_graph(x_data_mask, y_data_mask, 'local_volume', 'local_max', result_data, png, colors=colors_mask)

        x_data = np.array(result_data['local_max'])/np.array(result_data['TwoFoFc_std'])
        y_data = np.array(result_data['local_std'])/np.array(result_data['local_mean'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'local_max_2fofcstd_std_mean.png')
        print_graph(x_data_mask, y_data_mask, 'local_max/2FoFc_std', 'local_std/local_mean', result_data, png, colors=colors_mask)

        x_data = np.array(result_data['local_volume'])
        y_data = np.array(result_data['local_max'])/np.array(result_data['TwoFoFc_std'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_volume_max_2fofcstd.png')
        print_graph(x_data_mask, y_data_mask, 'local_volume', 'local_max/2FoFc_std', result_data, png, colors=colors_mask)

        x_data = np.array(result_data['FoFc_std'])
        y_data = np.array(result_data['TwoFoFc_std'])
        x_data_mask = x_data
        y_data_mask = y_data
        colors_mask = colors
        png = os.path.join(output_dir, 'CLUSTER_FoFc_2FoFo.png')
        print_graph(x_data_mask, y_data_mask, 'local_FoFc_std', 'local_2FoFo_std', result_data, png, colors=colors_mask)


        x_data = np.array(result_data['local_electrons'])/np.array(result_data['FoFc_std'])
        y_data = np.array(result_data['local_volume'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_electrons_fofcstd_volume.png')
        print_graph(x_data_mask, y_data_mask, 'local_electrons_FoFc_std', 'local_volume', result_data, png, colors=colors_mask)

        x_data = (np.array(result_data['local_electrons'])-np.array(result_data['local_volume'])*4*np.array(result_data['FoFc_std']))/np.array(result_data['FoFc_std'])
        y_data = np.array(result_data['local_volume'])
        x_data_mask = x_data[mask]
        y_data_mask = y_data[mask]
        colors_mask = colors[mask]
        png = os.path.join(output_dir, 'CLUSTER_local_electrons_upper_fofcstd_volume.png')
        print_graph(x_data_mask, y_data_mask, '[local_electrons - 4*local_volume*FoFc_std] / FoFc_std', 'local_volume', result_data, png, colors=colors_mask)

        x_data = (np.array(result_data['local_electrons'])-np.array(result_data['local_volume'])*4*np.array(result_data['FoFc_std']))/np.array(result_data['FoFc_std'])
        y_data = np.array(result_data['local_volume'])
        res_key = colors_dict['NAG']
        res_mask = colors == res_key
        x_data_mask = x_data[res_mask]
        y_data_mask = y_data[res_mask]
        #colors_mask = colors[res_mask]
        png = os.path.join(output_dir, 'NAG_local_electrons_upper_fofcstd_volume.png')
        print_graph(x_data_mask, y_data_mask, '[local_electrons - 4*local_volume*FoFc_std] / FoFc_std', 'local_volume', result_data, png)

    # ===========================
    if PARTITIONING_GRAPHS is True:

        f_name = os.path.join(file_dir, 'all_summary.txt')
        f_name_stub = os.path.basename(f_name).split('_')[0]
        global_data_dir = os.path.join(file_dir, f_name)
        result_data_keys = get_data_keys(global_data_dir)
        result_data = get_data(global_data_dir, result_data_keys)

        for ii in range(11-1):
            y_data = np.array(result_data['part_%02d_blob_parts' % (ii+1)])-np.array(result_data['part_%02d_blob_parts' % ii])
            plt.hist(y_data, 20, normed=1, label='%d - %d Fo-Fc std' % (ii+1, ii))

        fig_filename = os.path.join(output_dir, 'all_threshold_hist.png')
        plt.xlabel('part diff')
        plt.ylabel('%')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(fig_filename, dpi=150)
        plt.clf()

        for f_name in glob.iglob(os.path.join(file_dir, '*summary.txt')):
            f_name_stub = os.path.basename(f_name).split('_')[0]

            global_data_dir = os.path.join(file_dir, f_name)
            result_data_keys = get_data_keys(global_data_dir)
            result_data = get_data(global_data_dir, result_data_keys)

            x_data = np.arange(3, 14)

            for ii, res in enumerate(result_data['title']):
                y_ele = []
                y_part = []
                for i in range(11):
                    parts_key = 'part_%02d_blob_parts' % i
                    electron_key = 'part_%02d_blob_elecron_sum' % i

                    y_part.append(result_data[parts_key][ii])
                    y_ele.append(result_data[electron_key][ii])

                print_partitioning(x_data, y_ele, y_part)

            fig_filename = os.path.join(output_dir, '%s_ele_decy.png' % f_name_stub)
            plt.xlabel('Fo-Fc_std')
            plt.ylabel('blob electron sum')
            plt.title('%s' % f_name_stub)
            #plt.legend()
            plt.gcf().set_size_inches(15, 10)
            plt.savefig(fig_filename, dpi=150)
            plt.clf()

            for ii, res in enumerate(result_data['title']):
                y_vol = []
                y_part = []
                for i in range(11):
                    parts_key = 'part_%02d_map_blob_parts' % i
                    vol_key = 'part_%02d_map_blob_volume_sum' % i

                    y_part.append(result_data[parts_key][ii])
                    y_vol.append(result_data[vol_key][ii])

                print_partitioning(x_data, y_ele, y_part)

            fig_filename = os.path.join(output_dir, '%s_vol_decy.png' % f_name_stub)
            plt.xlabel('Fo-Fc_std')
            plt.ylabel('blob volume sum')
            plt.title('%s' % f_name_stub)
            #plt.legend()
            plt.gcf().set_size_inches(15, 10)
            plt.savefig(fig_filename, dpi=150)
            plt.clf()

    if CORRELATION_SHAPE is True:
        global_data_dir = os.path.join(data_dir, 'all_summary.txt')
        result_data = pd.read_csv(global_data_dir, sep=';', header=0, na_values=['n/a', 'nan'])
        print global_data_dir

        result_data = result_data.drop(['title', 'pdb_code', 'res_id', 'chain_id',
                             'fo_col', 'fc_col', 'weight_col', 'grid_space',
                             'solvent_radius', 'solvent_opening_radius',
                             'part_step_FoFc_std_min', 'part_step_FoFc_std_max',
                             'part_step_FoFc_std_step',
                             # for regression
                             #'local_res_atom_count', 'local_res_atom_occupancy_sum',
                             #'local_res_atom_electron_sum', 'local_res_atom_electron_occupancy_sum',
                             # quality
                             'local_BAa', 'local_NPa', 'local_Ra', 'local_RGa',
                             'local_SRGa', 'local_CCSa', 'local_CCPa', 'local_ZOa',
                             'local_ZDa', 'local_ZD_minus_a', 'local_ZD_plus_a',
                             #'solvent_mask_count', 'modeled_mask_count', 'void_mask_count',
                             ], axis=1)

        res_name_count = result_data['res_name'].value_counts()
        result_data['res_name_count'] = result_data['res_name'].map(res_name_count).astype(int)
        result_data = result_data[result_data['res_name_count'] > 10]
        result_data = result_data[result_data['res_name_count'] < 500]
        result_data = result_data.drop(['res_name_count'],  axis=1)

        result_data2 = result_data.drop(['res_name'],  axis=1)
        f, ax = plt.subplots(figsize=(9, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.corrplot(result_data2, annot=False, sig_stars=False, diag_names=False, cmap=cmap, ax=ax)
        f.tight_layout()
        f.show()
        f.set_size_inches(100, 100)
        f.savefig('corr.png', dpi=100)
        #print result_data
        #sns.set()
        #sns.pairplot(result_data, hue="res_name", size=1.0)
        #sns.show()

        #g = sns.PairGrid(result_data, hue="res_name")
        #g.map_diag(plt.hist)
        #g.map_offdiag(plt.scatter)
        #g.add_legend()

        """
        corrmat = result_data.corr()
        sns.set(context="paper", font="monospace")

        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, linewidths=0, square=True)

        networks = corrmat.columns.get_level_values("network")
        for i, network in enumerate(networks):
            if i and network != networks[i - 1]:
                ax.axhline(len(networks) - i, c="w")
                ax.axvline(i, c="w")

        f.tight_layout()
        """
