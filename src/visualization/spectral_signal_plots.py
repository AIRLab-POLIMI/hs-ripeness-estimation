import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.signal import savgol_filter


def snv(input_data):
    """
    Applies Standard Normal Variate

        Parameters:
            input_data (np.array): data to correct

        Returns:
            output_data (np.array): corrected data
    """

    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):

        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])

    return output_data

def msc(input_data, reference=None):
    """
    Applies Multi Scattering Correction

        Parameters:
            input_data (np.array): data to correct
            reference (np.array): reference spectrum, defaults to the mean of the input data

        Returns:
            data_msc (np.array): corrected data
    """
    input_data_ = np.copy(input_data)

    # mean centre correction
    for i in range(input_data_.shape[0]):
        input_data_[i,:] -= input_data_[i,:].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data_, axis=0)
    else:
        ref = reference

    data_msc = np.zeros_like(input_data_)
    for i in range(input_data_.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data_[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data_[i,:] - fit[0][1]) / fit[0][0]
    return data_msc


target_path = '/home/user/data/interim/visualization/spectral_signal_plots/'
data_path = 'home/user/data/processed/'

selected_range = list(range(1,53))
bands = [500.0,508.2,510.0,516.3,524.5,532.7,540.8,549.0,556.5,557.1,565.3,573.5,581.6,589.8,598.0,600.0,606.1,614.3,622.4,630.6,638.8,646.9,655.1,663.3,671.4,679.6,687.8,695.9,704.1,712.2,720.4,728.6,736.7,744.9,753.1,761.2,769.4,777.6,785.7,793.9,802.0,810.2,818.4,826.5,834.7,842.9,851.0,859.2,867.3,875.5,883.7,891.8,900.0]
columns = [str(bands[e]) for e in selected_range]

df = pd.read_csv(data_path+'2021-09-06-bunches/dataset.csv')
plants = pd.read_excel('plants/plants_dataset.xlsx')

data = df.iloc[:,1:len(selected_range)+1]
data = np.array(data,float)
p_data = plants.iloc[:,1:len(selected_range)+1]
p_data = np.array(p_data,float)

data_ = data
new_range = selected_range
plt.figure(figsize=(7.8,4.8))
for i in range(df.shape[0]):
    plt.plot(new_range,data_[i,:])
plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((0,1))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig('bunch_signals.pdf',format='pdf')

data_ = p_data
new_range = selected_range
plt.figure(figsize=(7.8,4.8))
for i in range(plants.shape[0]):
    plt.plot(new_range,data_[i,:])
plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((0,1))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig('plants_signals.pdf',format='pdf')

b_min = np.min([np.min(df.Brix),np.min(plants.Brix)])
b_max = np.max([np.max(df.Brix),np.max(plants.Brix)])
a_min = np.min([np.min(df.Antociani),np.min(plants.Antociani)])
a_max = np.max([np.max(df.Antociani),np.max(plants.Antociani)])

wl = 8
ord = 7
der = 1
data_ = snv(data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
value = df.Brix
c_norm = mpl.colors.Normalize(vmin=b_min, vmax=b_max)
c_map  = mpl.cm.gnuplot
s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
s_map.set_array([])
plt.figure(figsize=(7.8,4.8))
for i in range(df.shape[0]):
    plt.plot(new_range,data_[i,:], color=s_map.to_rgba(value[i]))
plt.xticks(selected_range[::6],columns[::6])
plt.colorbar(s_map, label='TSS (°Brix)')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('First derivative')
plt.ylim((-0.3,0.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'TSS_bunches.pdf',format='pdf')

wl = 15
ord = 4
der = 2
data_ = snv(p_data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
value = plants.Brix
c_norm = mpl.colors.Normalize(vmin=b_min, vmax=b_max)
c_map  = mpl.cm.gnuplot
s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
s_map.set_array([])
plt.figure(figsize=(7.8,4.8))
for i in range(plants.shape[0]):
    plt.plot(new_range,data_[i,:], color=s_map.to_rgba(value[i]))
plt.xticks(selected_range[::6],columns[::6])
plt.colorbar(s_map, label='TSS (°Brix)')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Second derivative')
plt.ylim((-0.15,0.15))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'TSS_plants.pdf',format='pdf')

wl = 15
ord = 4
der = 2
data_ = snv(p_data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
plt.figure(figsize=(7.8,4.8))
for i in range(plants.shape[0]):
    plt.plot(new_range,data_[i,:], color='red'if plants.Date[i] == 'sep' else 'blue')
plt.xticks(selected_range[::6],columns[::6])
red_patch = mpatches.Patch(color='red', label='September 2021')
blue_patch = mpatches.Patch(color='blue', label='August 2021')
plt.legend(handles=[red_patch,blue_patch])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Second derivative')
plt.ylim((-0.15,0.15))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'TSS_plants_datediff.pdf',format='pdf')

wl = 14
ord = 9
der = 1
data_ = snv(data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
value = df.Antociani
c_norm = mpl.colors.Normalize(vmin=a_min, vmax=a_max)
c_map  = mpl.cm.gnuplot
s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
s_map.set_array([])
plt.figure(figsize=(7.8,4.8))
for i in range(df.shape[0]):
    plt.plot(new_range,data_[i,:], color=s_map.to_rgba(value[i]))
plt.xticks(selected_range[::6],columns[::6])
plt.colorbar(s_map, label='Anthocyans (mg/g)')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('First derivative')
plt.ylim((-0.3,0.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'Anth_bunches.pdf',format='pdf')

wl = 13
ord = 3
der = 2
data_ = snv(p_data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
value = plants.Antociani
c_norm = mpl.colors.Normalize(vmin=a_min, vmax=a_max)
c_map  = mpl.cm.gnuplot
s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
s_map.set_array([])
plt.figure(figsize=(7.8,4.8))
for i in range(plants.shape[0]):
    plt.plot(new_range,data_[i,:], color=s_map.to_rgba(value[i]))
plt.xticks(selected_range[::6],columns[::6])
plt.colorbar(s_map, label='Anthocyans (mg/g)')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Second derivative')
plt.ylim((-0.15,0.15))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'Anth_plants.pdf',format='pdf')


wl = 13
ord = 3
der = 2
data_ = snv(p_data)
data_ = savgol_filter(data_,window_length=wl,polyorder=ord,deriv=der)
to_exclude = wl//2
data_ = data_[:,to_exclude:-to_exclude]
new_range = selected_range[to_exclude:-to_exclude]
plt.figure(figsize=(7.8,4.8))
for i in range(plants.shape[0]):
    plt.plot(new_range,data_[i,:], color='red'if plants.Date[i] == 'sep' else 'blue')
plt.xticks(selected_range[::6],columns[::6])
red_patch = mpatches.Patch(color='red', label='September 2021')
blue_patch = mpatches.Patch(color='blue', label='August 2021')
plt.legend(handles=[red_patch,blue_patch])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Second derivative')
plt.ylim((-0.15,0.15))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'Anth_plants_datediff.pdf',format='pdf')

value = list(range(data.shape[0]))
c_norm = mpl.colors.Normalize(vmin=np.min(value), vmax=np.max(value))
c_map  = mpl.cm.viridis
s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
s_map.set_array([])
plt.figure(figsize=(7.8,4.8))
for i in range(df.shape[0]):
    plt.plot(selected_range,data[i,:], color=s_map.to_rgba(value[i]))
    plt.xticks(selected_range[::6],columns[::6])
plt.colorbar(s_map, label='Acquisition Time')
plt.tight_layout()
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((0,0.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'acquisition_times.pdf',format='pdf')

plt.figure()
for i in range(df.shape[0]):
    color = 'red' if int(df.id[i][3:]) < 14 else 'blue'
    plt.plot(selected_range, data[i,:], color=color)
    plt.xticks(selected_range[::6],columns[::6])
patch_first = mpatches.Patch(color='red',label='First row')
patch_second = mpatches.Patch(color='blue',label='Second row')
plt.legend(handles=[patch_first,patch_second])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((0,0.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'row_division.pdf',format='pdf')

temp = snv(data)
plt.figure()
for i in range(df.shape[0]):
    plt.plot(selected_range, temp[i,:])
    plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((-1.7,1.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'snv.pdf',format='pdf')

temp = msc(data)
plt.figure()
for i in range(df.shape[0]):
    plt.plot(selected_range, temp[i,:])
    plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((-0.25,0.25))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'msc.pdf',format='pdf')

from scipy.signal import savgol_filter
temp = snv(data)
sg = savgol_filter(temp,6,3,0)
plt.figure()
for i in range(df.shape[0]):
    plt.plot(selected_range, sg[i,:])
    plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((-1.7,1.7))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'sg_smooth.pdf',format='pdf')

temp = snv(data)
sg = savgol_filter(temp,6,3,2)
to_exclude = 3
plt.figure()
for i in range(df.shape[0]):
    plt.plot(selected_range, sg[i,:],'black')
    plt.plot(selected_range[to_exclude:-(to_exclude)], sg[i,to_exclude:-(to_exclude)])
    plt.xticks(selected_range[::6],columns[::6])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.ylim((-0.3,0.3))
plt.xlim((-1,53))
plt.tight_layout()
plt.savefig(target_path+'sg_derivative.pdf',format='pdf')