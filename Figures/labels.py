# a stacked bar plot with errorbars
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Classification")))

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import util


cmb_model = util.load_model("../Data/cmb.pkl")
tamc_model = util.load_model("../Data/tamc.pkl")
cl_model = util.load_model("../Data/cl.pkl")


data_cmb = util.DatasetStatistics(cmb_model.data_frame, cmb_model.data_frame.loc[:, cmb_model.class_attribute]).classes.to_dict()
data_tamc = util.DatasetStatistics(tamc_model.data_frame, tamc_model.data_frame.loc[:, cmb_model.class_attribute]).classes.to_dict()
data_cl = util.DatasetStatistics(cl_model.data_frame, cl_model.data_frame.loc[:, cmb_model.class_attribute]).classes.to_dict()

filenames = ['SFig2_a.png', 'SFig2_b.png', 'SFig2_c.png']
datasets =  [data_cmb, data_tamc, data_cl]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
y_ranges = [55000, 40000, 45000]

subplot = 311
borderw = 1.0
borderh = 2.0
figw_base = (30.0-borderw)
figh_base = 7.0-borderw
figh = borderw+figh_base


for filename, data, color in zip(filenames, datasets, colors):
    data_sort = sorted(sorted(data.items(), key=lambda x:x[0]), key=lambda x:x[1], reverse=True)
    names = [_[0] for _ in data_sort]
    values = [_[1] for _ in data_sort]

    print names, values
    N = len(names)

    ind = np.arange(0, N)    # the x locations for the groups
    space = 0.3
    i = 2.0
    width = (1-space)       # the width of the bars: can also be len(x) sequence

    figw = borderw+(figw_base/200.0*N)
    fig = plt.figure(figsize=(figw, figh))
    ax = fig.add_subplot(111)

    pos = ind
    p1 = ax.bar(pos, values, width=width, align='center', color=color)

    plt.ylabel('Count')
    plt.xlabel('Label')
    ax.set_xlim(-width,N-1+width)
    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim(0, 52000)
    ax.set_yticks(np.arange(0, 52000, 5000))

    plt.subplots_adjust(left=(borderw-0.1)/figw, bottom=(borderh-0.1)/figh, right=1-0.1/figw, top=1-0.1/figh)
    # or 
    #plt.tight_layout()
    fig.savefig(filename, dpi=300)
    fig.savefig(filename.replace('png', 'svg'), dpi=300)
    fig.savefig(filename.replace('png', 'svg'), rasterize=False, dpi=300)
