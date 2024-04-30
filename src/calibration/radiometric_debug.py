from radiometric_calibration import mean_dark_curr, panel_values
import spectral.io.envi as envi
import glob
from tqdm import tqdm
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import re
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def radiometric_debug(cfg: DictConfig):

    reference_1_hdr = cfg.db.calibration.radiometric.name_hdr
    reference_1_dat = cfg.db.calibration.radiometric.name_dat
    wavelength_range_start = cfg.db.calibration.radiometric.wavelength_range_start
    wavelength_range_stop = cfg.db.calibration.radiometric.wavelength_range_stop
    panel_config = cfg.db.calibration.radiometric.panel_config
    panel_vals = cfg.db.calibration.radiometric.panel_vals
    dark_curr = cfg.db.calibration.radiometric.dark_curr
    dark_curr_target = cfg.db.calibration.radiometric.dark_curr_target
    calibration_time = cfg.db.calibration.radiometric.calibration_time
    bands = cfg.db.calibration.radiometric.year_bands
    target_folder = cfg.db.calibration.radiometric.debug_folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    debug_regression = target_folder + '/regression'
    if not os.path.exists(debug_regression):
        os.makedirs(debug_regression)
    debug_histograms = target_folder + '/histograms'
    if not os.path.exists(debug_histograms):
        os.makedirs(debug_histograms)
    exposure_diff = False

                        
    wavelength_range = list(range(wavelength_range_start, wavelength_range_stop))
    columns = [str(bands[e]) for e in wavelength_range]
    panel_config = np.array(panel_config)
    # if the calibration is done through a Mapir panel, recover the known reflectance values
    if isinstance(panel_vals, str):
        white, light_gray, dark_gray, black = panel_values(panel_vals, bands, diffuse=False)
    # else use the configuration known value
    else:
        panel_vals = np.array(panel_vals)
    # if there are dark current acquitions of the calibration phase
    if dark_curr != None:
        # compute the mean calibration dark current
        mean_dc = mean_dark_curr(dark_curr)
        # if there are dark current acquitions of the acquisition phase
        if dark_curr_target != None:
            # compute the mean acquisition dark current
            exposure_diff = True
            mean_dc_target = mean_dark_curr(dark_curr_target)
            dark_current = mean_dc_target
        else:
            dark_current = mean_dc

    # array for storing the results
    Results = np.zeros((len(wavelength_range), 2))
    # read hypercube
    spyFile = envi.open(reference_1_hdr, reference_1_dat)
    cal = np.array(spyFile.load(dtype=np.float32))
    # if it exists, correct the calibration hypercube using the mean dark current
    if dark_curr != None:
        cal = cal - mean_dc
    # if calibration and acquisition exposure times differ, divide by the calibration exposure time
    if exposure_diff:
        cal = cal/calibration_time
    # array storing the mean ROI values for each wavelength (row) and each panel section (col)
    DataSpectra = np.zeros((len(wavelength_range),len(panel_config)))

    # computation of the mean ROI value
    if isinstance(panel_vals, str):
        names = ['White', 'Light gray', 'Dark gray', 'Black']
        for k in wavelength_range:  # for each wavelength
            fig, ax = plt.subplots(2,2,sharex=True, sharey=True)
            ax = ax.flatten()
            for j in range(len(panel_config)):  # for each panel section
                ROI = cal[panel_config[j,0]:panel_config[j,1],panel_config[j,2]:panel_config[j,3],k]
                DataSpectra[k,j] = np.mean(ROI)
                f = ROI.flatten()
                sns.histplot(f,ax=ax[j])
                ax[j].set_ylim((0,750))
                ax[j].set_xlim((0,65535))
                ax[j].set_xlabel('Digital Number (DN)')
                ax[j].set_ylabel('Pixel count')
                ax[j].set_title(names[j])
                ax[j].grid()
            fig.tight_layout()
            fig.savefig(target_folder+'/histograms/band_'+columns[k]+'.png')
            plt.close()
    else:
        for k in wavelength_range:  # for each wavelength
            plt.figure()
            for j in range(len(panel_config)):  # for each panel section
                ROI = cal[panel_config[j,0]:panel_config[j,1],panel_config[j,2]:panel_config[j,3],k]
                DataSpectra[k,j] = np.mean(ROI)
                f = ROI.flatten()
                sns.histplot(f)
                plt.ylim((0,12000))
                plt.xlim((0,65535))
                plt.xlabel('Digital Number (DN)')
                plt.ylabel('Pixel count')
                plt.grid()
                plt.tight_layout()
                plt.savefig(target_folder+'/histograms/band_'+columns[k]+'.png')
                plt.close()
        
    # plot the calibration mean spectrum over the considered panels
    if len(panel_config) == 1:
        plt.figure()
        plt.plot(DataSpectra[:,0])
        plt.ylim((0,100000))
        plt.xlim((-1,53))
        plt.ylabel('Digital Number (DN)')
        plt.xlabel('Wavelength (nm)')
        plt.xticks(wavelength_range[::5],columns[::5])
        plt.grid()
        plt.tight_layout()
        plt.savefig(target_folder+'/calibration_spectrum.png')
        plt.close()
    else:
        colors = ['green','orange','red','blue']
        plt.figure()
        for j in range(len(panel_config)):
            plt.plot(DataSpectra[:,j], colors[j])
        plt.legend(['white','light gray', 'dark gray', 'black'])
        plt.ylim((0,100000))
        plt.xlim((-1,53))
        plt.ylabel('Digital Number (DN)')
        plt.xlabel('Wavelength (nm)')
        plt.xticks(wavelength_range[::5],columns[::5])
        plt.grid()
        plt.tight_layout()
        plt.savefig(target_folder+'/calibration_spectrum.png')
        plt.savefig(target_folder+'/calibration_spectrum.png')
        plt.close()

    # regression coefficients
    for k in wavelength_range:  # for each wavelength
        # empirical values
        x = np.array(DataSpectra[k,:])
        # known values
        if isinstance(panel_vals, str):
            y = np.array( [white[k],light_gray[k], dark_gray[k], black[k]])
        else: 
            y = panel_vals
            if len(panel_config) == 1:
                x = np.append(x,0.)
                y = np.append(y,0.)

        # linear regression
        model = LinearRegression().fit(x.reshape(-1,1),y.reshape(-1,1))
        Results[k,0] = model.intercept_
        Results[k,1] = model.coef_

    # regression plots
    xx = np.linspace(0,65535,100)
    if isinstance(panel_vals, str):
        for k in wavelength_range:
            plt.figure()
            plt.plot(xx,Results[k,0]+xx*Results[k,1],'-b')
            plt.plot(DataSpectra[k,0],white[k],'or')
            plt.plot(DataSpectra[k,1],light_gray[k],'or')
            plt.plot(DataSpectra[k,2],dark_gray[k],'or')
            plt.plot(DataSpectra[k,3],black[k],'or')
            plt.ylabel('Reflectance')
            plt.xlabel('Digital Number (DN)')
            plt.ylim((-0.1,1.02))
            plt.xlim((-300,65535))
            plt.grid()
            plt.tight_layout()
            plt.savefig(target_folder+'/regression/band_'+columns[k]+'.png')
            plt.close()
    else:
        for k in wavelength_range:
            plt.figure()
            plt.plot(xx,Results[k,0]+xx*Results[k,1],'-b')
            plt.plot(DataSpectra[k,0],panel_vals,'or')
            plt.ylabel('Reflectance')
            plt.xlabel('Digital Number (DN)')
            plt.ylim((-0.1,1.02))
            plt.xlim((-300,65535))
            plt.grid()
            plt.tight_layout()
            plt.savefig(target_folder+'/regression/band_'+columns[k]+'.png')
            plt.close()

if __name__ == '__main__':
    radiometric_debug()





