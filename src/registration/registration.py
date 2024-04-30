'''
Implementation and application of hyperspectral registration, to be applied on calibrated images.
Functions:
- psr_registration: intensity-based registration
- feature_sift_registration: feature-based (SIFT) registration
Main:
- registration: application of image registration
'''

# libraries
import numpy as np
from pystackreg import StackReg
import cv2 as cv
import spectral.io.envi as envi
import os
import glob
from tqdm import tqdm
import re
import hydra
from omegaconf import DictConfig
import SimpleITK as sitk

# function for intensity based refistration using pystackreg library
def psr_registration(img,
                     reference_band = 28,
                     selected_range = list(range(1,53))
                     ):
    '''
    Applies intensity-based registration through pystackreg over a single hypercube given
    a reference band

        Parameters:
                img (np.array): hypercube to be registered
                reference_band (int): index of the reference band
                selected_range (list): list of indexes of the bands to be registered
    '''

    sr = StackReg(StackReg.RIGID_BODY)
    # registration (intensity-based w.r.t. a reference band)
    for w in selected_range:  # for each wavelength register the considered image to the reference one
        if w != reference_band:
            img[:,:,w] = sr.register_transform(img[:,:,reference_band].reshape((img.shape[0],img.shape[1])), img[:,:,w].reshape((img.shape[0],img.shape[1])))
    
def simpleitk_registration(img,
                           reference_band = 28,
                           selected_range = list(range(1,53)),
                           max_shift = 200,
                           samples_per_axis = 10,
                           n_histogram_bins = 100,
                           convergence_window_size = 10,
                           convergence_min_value = 1e-6,
                           n_iterations = 100
                           ):

    fixed_image = sitk.GetImageFromArray(img[:,:,reference_band])
    for w in selected_range:
        if w != reference_band:
            moving_image = sitk.GetImageFromArray(img[:,:,w])

            # initialization

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_histogram_bins)
            registration_method.SetMetricSamplingStrategy(registration_method.NONE)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsExhaustive(
                numberOfSteps=[0, samples_per_axis//2, samples_per_axis//2]
            )
            registration_method.SetOptimizerScales([1, max_shift//samples_per_axis, max_shift//samples_per_axis])
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            registration_method.Execute(fixed_image, moving_image)

            # registration

            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_histogram_bins)
            registration_method.SetMetricSamplingStrategy(registration_method.NONE)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1,
                numberOfIterations=n_iterations,
                convergenceMinimumValue=convergence_min_value,
                convergenceWindowSize=convergence_window_size,
                estimateLearningRate=sitk.ImageRegistrationMethod.EachIteration
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = registration_method.Execute(fixed_image, moving_image)
            moving_resampled = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID(),
            )

            img[:,:,w] = sitk.GetArrayFromImage(moving_resampled)

    
@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def registration(cfg: DictConfig):
    '''
    Application of the intensity-based registration procedure. The registered hypercubes (envi format) 
    and images related to extreme and false RGB bands are saved.
    '''

    # general
    data_dir = cfg.db.registration.general.source_folder
    target_folder = cfg.db.registration.general.target_folder
    batch_size = cfg.db.registration.general.batch_size
    selected_range_start = cfg.db.registration.general.selected_range_start
    selected_range_stop = cfg.db.registration.general.selected_range_stop
    selected_range = list(range(selected_range_start,selected_range_stop))
    false_RGB = cfg.db.registration.general.false_RGB
    # intensity based registration
    do_psr = cfg.db.registration.intensity.do_psr
    ref_band_ib = cfg.db.registration.intensity.ref_band
    # simpleITK registration
    do_sitk = cfg.db.registration.sitk.do_sitk
    ref_band_sitk = cfg.db.registration.sitk.ref_band_sitk
    max_shift = cfg.db.registration.sitk.max_shift
    samples_per_axis = cfg.db.registration.sitk.samples_per_axis
    n_histogram_bins = cfg.db.registration.sitk.n_histogram_bins
    convergence_window_size = cfg.db.registration.sitk.convergence_window_size
    convergence_min_value = cfg.db.registration.sitk.convergence_min_value
    n_iterations = cfg.db.registration.sitk.n_iterations

    # data location
    dir_files = glob.glob(data_dir + '/*.dat')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    false_RGB_folder = target_folder+'/false_RGB'
    if not os.path.exists(false_RGB_folder):
        os.makedirs(false_RGB_folder)
    extremes_folder = target_folder+'/extremes'
    if not os.path.exists(extremes_folder):
        os.makedirs(extremes_folder)

    # create batches (possibly reduces overhead)
    batches = []
    full_batches = len(dir_files)//batch_size
    rest = len(dir_files)%batch_size
    for i in range(full_batches):
        batches.append(dir_files[i*batch_size:(i+1)*batch_size])
    if rest > 0:
        batches.append(dir_files[(full_batches*batch_size):])

    # plot bands
    extremes = [selected_range[-1],selected_range[len(selected_range)//2],selected_range[0]]

    # for every batch
    for batch in tqdm(batches):

        filenames = []
        imgsArray = []

        # load files
        for fname in batch:

            full_name = fname[:-4]
            name = re.split('/',full_name)[-1]
            filenames.append(name)
            spyFile = envi.open(full_name+'.hdr',fname)
            imgsArray.append(np.array(spyFile.load(dtype=np.float32)))

        
        for i in range(len(imgsArray)):
            
            # registration         
            if do_psr:
                psr_registration(img = imgsArray[i], reference_band = ref_band_ib, selected_range = selected_range)

            if do_sitk:
                simpleitk_registration(img = imgsArray[i], reference_band = ref_band_sitk, selected_range = selected_range,
                                       max_shift = max_shift, samples_per_axis = samples_per_axis, n_histogram_bins = n_histogram_bins,
                                       convergence_window_size = convergence_window_size, convergence_min_value = convergence_min_value,
                                       n_iterations = n_iterations)

            # save + plots
            envi.save_image(hdr_file=target_folder+'/'+filenames[i]+'.hdr', image=imgsArray[i], ext='.dat',interleave='bsq', force=True)
            cv.imwrite(false_RGB_folder+'/'+filenames[i]+'.png' ,255*imgsArray[i][:,:,false_RGB])
            cv.imwrite(extremes_folder+'/'+filenames[i]+'.png' ,255*imgsArray[i][:,:,extremes])


if __name__ == '__main__':

    registration()


