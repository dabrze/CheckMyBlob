#!/usr/bin/env python

import numpy as np

from skimage import measure
from skimage import morphology

from scipy.optimize import curve_fit
from scipy.stats import norm

try:
    MATPLOTLIB = True
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except Exception as err:
    print "MATPLOT LIB IMPOR ERR"
    print err
    MATPLOTLIB = False


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def binormal(x, m1, s1, m2, s2, scale=0.5):
    norm1 = norm(loc=m1, scale=s1)
    norm2 = norm(loc=m2, scale=s2)
    return scale*norm1.pdf(x) + (1.0-scale)*norm2.pdf(x)


def normal(x, m1, s1):
    norm1 = norm(loc=m1, scale=s1)
    return norm1.pdf(x)


def fit_binormal(x_bins, y_value, p0=None):
    try:
        popt, pcov = curve_fit(binormal, x_bins, y_value, p0=p0)
    except:
        return 5*[np.nan], np.nan
    return popt, pcov


def fit_normal(x_bins, y_value, p0=None):
    popt, pcov = curve_fit(normal, x_bins, y_value, p0=p0)
    return popt, pcov


def print_slices(cut_map_orth, mask, label_im, nb_labels):
    for ii in range(3):
        cut_map_slice = (ii + 1) * len(cut_map_orth) / 4
        plt.subplot('33%d' % (3 * ii + 1))
        plt.imshow(cut_map_orth[cut_map_slice])
        plt.axis('off')
        plt.subplot('33%d' % (3 * ii + 2))
        plt.imshow(mask[cut_map_slice], cmap=plt.cm.gray)
        plt.axis('off')
        plt.subplot('33%d' % (3 * ii + 3))
        plt.imshow(label_im[cut_map_slice], vmin=0, vmax=nb_labels, cmap=plt.cm.spectral)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
    plt.show()


def print_3d_mesh(cut_map_orth, diff_std, alpha1=0.1, map2=None, th2=None, alpha2=0.1, blob=None, map_min_o=None,
                  special_points=None, title="", logger=None, grid_space=0.2, morphology_skel3d=False):

    def show_scatter(ax, points, map_min_o, color, size):
        x = []
        y = []
        z = []
        for point in points:
            x.append((point[0] - map_min_o[0]))
            y.append((point[1] - map_min_o[1]))
            z.append((point[2] - map_min_o[2]))
        ax.scatter(x, y, z, c=color, s=size)

    if MATPLOTLIB is False:
        print "MATPLOTLIB not imported"
        return
    if np.nanmin(cut_map_orth) < diff_std < np.nanmax(cut_map_orth):
        if morphology_skel3d:
            skel = morphology.skeletonize_3d(cut_map_orth > diff_std)
            if np.nanmin(skel) < 0.5 < np.nanmax(skel):
                map2 = skel
                th2 = 0.5

        if map2 is not None:
            arr1 = (cut_map_orth, map2)
            arr2 = (diff_std, th2)
            arr3 = (alpha1, alpha2)
            color = ([0.0, 0.0, 1.0, alpha1], [1.0, 0.0, 0.0, alpha2])
        else:
            arr1 = (cut_map_orth),
            arr2 = (diff_std),
            arr3 = (alpha1),
            color = ([0.0, 0.0, 1.0, alpha1]),
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        for cut_map_orth, global_std, _color, alpha in zip(arr1, arr2, color, arr3):
            try:
                scale = 1
                verts, faces = measure.marching_cubes(
                    cut_map_orth[::scale, ::scale, ::scale],
                    diff_std,
                    (scale * grid_space, scale * grid_space, scale * grid_space)
                )
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=_color, edgecolor=_color, shade=True)
            except:
                title = "NO MAP 2 !!! " + title

            if blob:
                show_scatter(ax, blob.local_maxi_o, map_min_o, [0.0, 1.0, 0.0, 1.0], 25)
                show_scatter(ax, blob.max_point_box_o_list, map_min_o, [1.0, 0.0, 0.0, 1.0], 55)

            if special_points:
                show_scatter(ax, special_points, map_min_o, [0.0, 0.0, 0.0, 1.0], 95)

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_xlim(0, grid_space * cut_map_orth.shape[0])
        ax.set_ylim(0, grid_space * cut_map_orth.shape[1])
        ax.set_zlim(0, grid_space * cut_map_orth.shape[2])
        ax.set_title(title)
        plt.show()
    else:
        if logger:
            logger.info("MAP NOT SHOWED %s %s %s" % (np.nanmin(cut_map_orth), diff_std, np.nanmax(cut_map_orth)))
