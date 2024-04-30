import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

target_path = '/home/user/data/interim/visualization/boxplots/'
data_path = 'home/user/data/processed/'

selected_range = list(range(1,53))

df = pd.read_csv(data_path+'2021-09-06-bunches/dataset.csv')
plants = pd.read_excel('plants/plants_dataset.xlsx')

sep = plants[plants['Date'] == 'sep']
aug = plants[plants['Date'] == 'aug']
sep_b = sep.Brix
aug_b = aug.Brix
sep_a = sep.Antociani
aug_a = aug.Antociani
props = dict(boxstyle='round', facecolor='white', alpha=1)
labels = ['August 2021','September 2021']
colors = ['lightblue','lightblue']
fig, ax = plt.subplots(1,2)
bplot = ax[0].boxplot([aug_b,sep_b], vert=True,patch_artist=True,labels=labels)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
textstr = '\n'.join((
    r'$\mathrm{August}$',
    r'$\mathrm{mean}:%.2f$' % (np.mean(aug_b), ),
    r'$\mathrm{std}:%.2f$' % (np.std(aug_b), ),
    r'$\mathrm{September}$',
    r'$\mathrm{mean}:%.2f$' % (np.mean(sep_b), ),
    r'$\mathrm{std}:%.2f$' % (np.std(sep_b), )))
ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax[0].grid(axis='y')
ax[0].set_ylabel('TSS (°Brix)')
ax[0].set_ylim((8,22))
bplot = ax[1].boxplot([aug_a,sep_a], vert=True,patch_artist=True,labels=labels)
colors = ['pink','pink']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
textstr = '\n'.join((
    r'$\mathrm{August}$',
    r'$\mathrm{mean}:%.2f$' % (np.mean(aug_a), ),
    r'$\mathrm{std}:%.2f$' % (np.std(aug_a), ),
    r'$\mathrm{September}$',
    r'$\mathrm{mean}:%.2f$' % (np.mean(sep_a), ),
    r'$\mathrm{std}:%.2f$' % (np.std(sep_a), )))
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax[1].grid(axis='y')
ax[1].set_ylabel('Anthocyanins (mg/g)')
ax[1].set_ylim((0,0.3))
plt.tight_layout()
plt.savefig(target_path+'boxplot_plants.pdf',format='pdf')

sep = df
sep = sep.dropna()
sep_b = sep.Brix
sep_a = sep.Antociani
labels = ['September 2021']
colors = ['lightblue']
props = dict(boxstyle='round', facecolor='white', alpha=1)
fig, ax = plt.subplots(1,2)
bplot = ax[0].boxplot([sep_b], vert=True,patch_artist=True,labels=labels,widths = 0.08)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
textstr = '\n'.join((
    r'$\mathrm{mean}:%.2f$' % (np.mean(sep_b), ),
    r'$\mathrm{std}:%.2f$' % (np.std(sep_b), )))
ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax[0].grid(axis='y')
ax[0].set_ylabel('TSS (°Brix)')
ax[0].set_ylim((8,22))
bplot = ax[1].boxplot([sep_a], vert=True,patch_artist=True,labels=labels,widths = 0.08)
colors = ['pink',]
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
textstr = '\n'.join((
    r'$\mathrm{mean}:%.2f$' % (np.mean(sep_a), ),
    r'$\mathrm{std}:%.2f$' % (np.std(sep_a), )))
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax[1].grid(axis='y')
ax[1].set_ylabel('Anthocyanins (mg/g)')
ax[1].set_ylim((0,0.3))
plt.tight_layout()
plt.savefig(target_path+'boxplot_bunches.pdf',format='pdf')