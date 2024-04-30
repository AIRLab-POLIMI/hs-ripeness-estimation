from calibration.calibration import Calibration
#from .calibration import Calibration
import numpy as np
import spectral.io.envi as envi
from sklearn.linear_model import LinearRegression
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import pandas as pd


def mean_dark_curr(dark_curr):
    """
    Computes the mean dark current given a folder containing multiple dark current acquisitions.

        Parameters:
            dark_curr (str): folder containing the acquisitions

        Returns:
            mean (np.array): lmean dark current hypercube
    """

    # open the dark current hypercubes in a list
    hdrs = glob.glob(dark_curr+'/*/*.hdr')
    imgsArray = []
    for hdr in hdrs:
        dat = hdr[:-4]+'.dat'
        spyFile = envi.open(hdr, dat)
        imgsArray.append(np.array(spyFile.load(dtype=np.float32)))
    # compute and return the mean
    mean = np.mean(imgsArray, axis = 0)
    return mean

def panel_values(panel_vals, wavelengths, diffuse = False):
    """
    Given a list of wavelengths, returns the corresponding (known) reflectance values of the Mapir panel.

        Parameters:
            panel_vals (str): folder containing the Mapir reflectance values
            wavelengths (list): list of considered wavelengths
            diffuse (bool): if True, use the diffuse reflectance values, else use the total ones (defaults to False)

        Returns:
            w (list): reflectance over the considered wavelengths of the white panel
            lg (list): reflectance over the considered wavelengths of the light gray panel
            dg (list): reflectance over the considered wavelengths of the dark gray panel
            b (list): reflectance over the considered wavelengths of the black panel
    """

    df = pd.read_excel(panel_vals)
    bands = [round(e) for e in wavelengths]
    b_tot = [df[df['Wavelength'] == e]['B total'].values[0]/100  for e in bands]
    b_diff = [df[df['Wavelength'] == e]['B diffuse'].values[0]/100  for e in bands]
    dg_tot = [df[df['Wavelength'] == e]['DG total'].values[0]/100  for e in bands]
    dg_diff = [df[df['Wavelength'] == e]['DG diffuse'].values[0]/100  for e in bands]
    lg_tot = [df[df['Wavelength'] == e]['LG total'].values[0]/100  for e in bands]
    lg_diff = [df[df['Wavelength'] == e]['LG diffuse'].values[0]/100  for e in bands]
    w_tot = [df[df['Wavelength'] == e]['W total'].values[0]/100  for e in bands]
    w_diff = [df[df['Wavelength'] == e]['W diffuse'].values[0]/100  for e in bands]
    if not(diffuse):
        return w_tot, lg_tot, dg_tot, b_tot
    else:
        return w_diff, lg_diff, dg_diff, b_diff


class RadiometricCalibration(Calibration):
    def __init__(self):
        super().__init__()
        self.dark_current = None
        self.exposure_diff = False

    def compute_params(self, reference_1_hdr, reference_1_dat, reference_2_hdr, reference_2_dat,
                       wavelength_range_start, wavelength_range_stop,
                       panel_config, panel_vals, dark_curr, dark_curr_target,
                       calibration_time, bands,
                       sep_threshold_band, sep_scaling, sep_hdr, sep_dat):
        """
        Computes the radiometric calibration coefficients given one (or two) calibration images.

            Parameters:
                reference_1_hdr (str): name of the header file of the image for calibration
                reference_1_dat (str): name of the data file of the image for calibration
                reference_2_hdr (str): (optional) name of the header file of the second image for calibration
                reference_2_dat (str): (optional) name of the data file of the second image for calibration
                wavelength_range_start (int): start of the considered band range
                wavelength_range_end (int): end of the considered band range
                panel_config (np.array): ROI boundaries of the panel (each row corresponds to a different panel section)
                panel_vals (list(list) or str): panel known value. If 'str', path to the folder containing the known Mapir values
                dark_curr (str): folder containing the dark current acquisitions of the calibration phase. 
                                 If dark_curr_target=None, the camera exposure time of calibration and acquisition 
                                 phases is the same and this folder is used for both the dark current corrections
                dark_curr_target (str): folder containing the dark current acquisitions of the acquisition phase. If None,
                                        the calibration dark current is used for target dark current correction
                calibration_time (int): exposure time of the calibration phase
                bands (list): considered wavelengths, used to recover Mapir panel values when needed
                sep_threshold_band (int): band from which, due to saturation, the calibration panel of September 2021 is considered
                sep_scaling (float): scaling applied to the the calibration panel of September 2021
                sep_hdr (str): name of the header file of the image containing the calibration panel of September 2021 
                sep_dat (str): name of the data file of the image containing the calibration panel of September 2021 
            
        """

        # variable initialization
        wavelength_range = list(range(wavelength_range_start, wavelength_range_stop))
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
                self.exposure_diff = True
                mean_dc_target = mean_dark_curr(dark_curr_target)
                self.dark_current = mean_dc_target
            else:
                self.dark_current = mean_dc
    
        # array for storing the results
        Results = np.zeros((len(wavelength_range), 2))
        # read hypercube
        spyFile = envi.open(reference_1_hdr, reference_1_dat)
        cal = np.array(spyFile.load(dtype=np.float32))
        # if it exists, correct the calibration hypercube using the mean dark current
        if dark_curr != None:
            cal = cal - mean_dc
        # if calibration and acquisition exposure times differ, divide by the calibration exposure time
        if self.exposure_diff:
            cal = cal/calibration_time
        # array storing the mean ROI values for each wavelength (row) and each panel section (col)
        DataSpectra = np.zeros((len(wavelength_range),len(panel_config)))

        # computation of the mean ROI value
        # reshape for coherent cycling if a single panel section is considered (2021)
        for k in wavelength_range:  # for each wavelength
            for j in range(len(panel_config)):  # for each panel section
                ROI = cal[panel_config[j,0]:panel_config[j,1],panel_config[j,2]:panel_config[j,3],k]
                DataSpectra[k,j] = np.mean(ROI)
            
        # if a second calibration image is given, repeat procedure with the second image
        if reference_2_hdr != None:  
            spyFile = envi.open(reference_2_hdr, reference_2_dat)
            cal_2 = np.array(spyFile.load(dtype=np.float32))
            if dark_curr != None:
                cal_2 = cal_2 - mean_dc
            if self.exposure_diff:
                cal_2 = cal_2/calibration_time
            DataSpectra_2 = np.zeros((len(wavelength_range),len(panel_config)))
            for k in wavelength_range:  
                for j in range(len(panel_config)):  
                    ROI = cal_2[panel_config[j,0]:panel_config[j,1],panel_config[j,2]:panel_config[j,3],k]
                    DataSpectra_2[k,j] = np.mean(ROI)
            # consider as final values the mean values bewteen the two images
            for k in wavelength_range:  
                for j in range(len(panel_config)):
                    DataSpectra[k,j] = np.mean([DataSpectra[k,j],DataSpectra_2[k,j]])

        # if the calibration panel of september 2021 is used
        if sep_hdr != None:
            spyFile = envi.open(sep_hdr, sep_dat)
            cal_sep = np.array(spyFile.load(dtype=np.float32))
            sep = np.zeros((len(wavelength_range),len(panel_config)))
            for k in wavelength_range:  
                for j in range(len(panel_config)):  
                    ROI = cal_sep[panel_config[j,0]:panel_config[j,1],panel_config[j,2]:panel_config[j,3],k]
                    sep[k,j] = (1/sep_scaling)*np.mean(ROI)

            temp = np.zeros((len(wavelength_range),len(panel_config)))
            for k in wavelength_range:
                for j in range(len(panel_config)):
                    if k < sep_threshold_band:
                        temp[k,j] = sep[k,j]
                    else:
                        temp[k,j] = DataSpectra[k,j]
            DataSpectra = temp
        
        # plot the calibration mean spectrum over the considered panels
        if len(panel_config) == 1:
            plt.figure()
            plt.plot(DataSpectra[:,0])
            plt.savefig('calibration_spectrum.png')
        else:
            colors = ['green','orange','red','blue']
            plt.figure()
            for j in range(len(panel_config)):
                plt.plot(DataSpectra[:,j], colors[j])
            plt.legend(['white','light gray', 'dark gray', 'black'])
            plt.savefig('calibration_spectrum.png')

        # regression coefficients
        for k in wavelength_range:  # for each wavelength
            # empirical values
            x = np.array(DataSpectra[k,:]) # all mapir sections
            #x = np.array(DataSpectra[k,1:]) # only non-saturated mapir sections
            # known values
            if isinstance(panel_vals, str):
                y = np.array([white[k], light_gray[k], dark_gray[k], black[k]]) # all mapir sections
                #y = np.array( [light_gray[k], dark_gray[k], black[k]]) # only non-saturated mapir sections
            else: 
                y = panel_vals
                if len(panel_config) == 1:
                    x = np.append(x,0.)
                    y = np.append(y,0.)

            # linear regression
            model = LinearRegression().fit(x.reshape(-1,1),y.reshape(-1,1))
            Results[k,0] = model.intercept_
            Results[k,1] = model.coef_

        self.params = Results

    def load_params(self, filepath):
        '''
        Loads the radiometric calibration coefficients given the file
        (.txt) containing them

            Parameters:
                    filepath (str): filepath (str): path to the folder in which the file is saved
        '''

        self.params = np.loadtxt(filepath, delimiter=",")

    def save_params(self, filepath):
        '''
        Saves the radiometric calibration coefficients in a specified folder (.txt)

            Parameters:
                    filepath (str): path to the folder in which the files are saved
        '''

        np.savetxt(filepath,self.params,delimiter=",")

    def apply_calibration(self, wavelength_range_start, wavelength_range_stop, scale, threshold, target_time, data):
        """
        Applies the radiometric calibration to a list of images using object parameters

            Parameters:
                data (list): list of images on which the geometric calibration is performed
                wavelength_range_start (int): first index of the considered band range
                wavelength_range_stop (int): last index of the considered band range
                scale (float): coefficient accounting for shadowing effects in the acquisition
                threshold (bool): flag indicating the thresholding to 1 of all reflectances > 1
                                  and to 0 of all reflectances < 1
        """

        wavelength_range = list(range(wavelength_range_start, wavelength_range_stop))

        for img in data:

            # dark current correction (using mean calibration dark current if the exposure times
            # of the acquisition phase are the same of the acquisition phase, else using the 
            # mean target dark current)
            if self.dark_current is not None:
                img -= self.dark_current
            # if calibration and acquisition exposure times differ, divide by the target exposure time
            if self.exposure_diff:
                img /= target_time
            # calibrated image
            for k in wavelength_range:  # for each wavelength apply calibration and scale correction
                img[:, :, k] = (img[:, :, k]*self.params[k, 1] + self.params[k, 0])*scale
            # threshold reflectance values to 1
            if threshold:
                img[img > 1] = 1
                img[img < 0] = 0
                
