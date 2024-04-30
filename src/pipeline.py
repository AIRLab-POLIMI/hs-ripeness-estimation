'''
Application the calibration pipeline, that is:
- Radiometric calibration
- Geometric calibration
The calibrated hypercubes are saved in envi format.
References: 
- src/calibration/calibration.py 
- src/calibration/radiometric_calibration.py
- src/calibration/geometric_calibration.py
'''

# libraries
# calibration
from calibration.geometric_calibration import GeometricCalibration
from calibration.radiometric_calibration import RadiometricCalibration
# registration
from registration.registration import registration
# segmentation
from segmentation.segmentation import segmentation
# general
import spectral.io.envi as envi
import glob
from tqdm import tqdm
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import re
import cv2 as cv


@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def calibration(cfg: DictConfig):

    print('Date: ' + str(cfg.db.date))
    source_folder = cfg.db.calibration.source_folder
    target_folder = cfg.db.calibration.target_folder
    batch_size = cfg.db.calibration.batch_size

    # data location
    dir_files = glob.glob(source_folder + '/*')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    cal_false_RGB_folder = target_folder + '/false_RGB'
    if not os.path.exists(cal_false_RGB_folder):
        os.makedirs(cal_false_RGB_folder)
    cal_extremes_folder = target_folder + '/extremes'
    if not os.path.exists(cal_extremes_folder):
        os.makedirs(cal_extremes_folder)
    false_RGB = cfg.db.registration.general.false_RGB
    selected_range_start = cfg.db.registration.general.selected_range_start
    selected_range_stop = cfg.db.registration.general.selected_range_stop
    selected_range = list(range(selected_range_start,selected_range_stop))
    extremes = [selected_range[-1],selected_range[len(selected_range)//2],selected_range[0]]


    # create batches
    batches = []
    full_batches = len(dir_files) // batch_size
    rest = len(dir_files) % batch_size
    for i in range(full_batches):
        batches.append(dir_files[i*batch_size:(i+1)*batch_size])
    if rest > 0:
        batches.append(dir_files[(full_batches*batch_size):])

    # Radiometric calibrator
    radiom_calibration = RadiometricCalibration()
    radiom_calibration.compute_params(cfg.db.calibration.radiometric.name_hdr,
                                      cfg.db.calibration.radiometric.name_dat,
                                      cfg.db.calibration.radiometric.second_hdr,
                                      cfg.db.calibration.radiometric.second_dat,
                                      cfg.db.calibration.radiometric.wavelength_range_start,
                                      cfg.db.calibration.radiometric.wavelength_range_stop,
                                      cfg.db.calibration.radiometric.panel_config,
                                      cfg.db.calibration.radiometric.panel_vals,
                                      cfg.db.calibration.radiometric.dark_curr,
                                      cfg.db.calibration.radiometric.dark_curr_target,
                                      cfg.db.calibration.radiometric.calibration_time,
                                      cfg.db.calibration.radiometric.year_bands,
                                      cfg.db.calibration.radiometric.sep_threshold_band,
                                      cfg.db.calibration.radiometric.sep_scaling,
                                      cfg.db.calibration.radiometric.sep_hdr,
                                      cfg.db.calibration.radiometric.sep_dat
                                      )

    # Geometric calibrator
    geom_calibration = GeometricCalibration()
    geom_calibration.load_params(cfg.db.calibration.geometric.distortion_1_name,
                                 cfg.db.calibration.geometric.distortion_2_name,
                                 cfg.db.calibration.geometric.intrinsic_matrix_1_name,
                                 cfg.db.calibration.geometric.intrinsic_matrix_2_name)

    for batch in tqdm(batches):
    
        filenames = []
        imgsArray = []

        for fname in batch:

            hdr = glob.glob(fname + '/*'+'.hdr')[0]
            dat = glob.glob(fname + '/*'+'.dat')[0]
            spyFile = envi.open(hdr, dat)
            imgsArray.append(np.array(spyFile.load(dtype=np.float32)))

            name = ''.join(re.findall('\d', re.split('/', fname)[-1]))
            filenames.append(name)

        # Radiometric calibration
        radiom_calibration.apply_calibration(cfg.db.calibration.radiometric.wavelength_range_start,
                                             cfg.db.calibration.radiometric.wavelength_range_stop,
                                             cfg.db.calibration.radiometric.scale,
                                             cfg.db.calibration.radiometric.threshold,
                                             cfg.db.calibration.radiometric.target_time,
                                             imgsArray)

        # Geometric calibration
        geom_calibration.apply_calibration(cfg.db.calibration.geometric.selected_range_start,
                                           cfg.db.calibration.geometric.selected_range_stop,
                                           cfg.db.calibration.geometric.sensor_boundary,
                                           imgsArray)

        # save
        for i in range(len(imgsArray)):
            envi.save_image(hdr_file=target_folder+'/sequence_'+filenames[i]+'.hdr', image=imgsArray[i], ext='.dat', interleave='bsq', force=True)
            cv.imwrite(cal_false_RGB_folder+'/sequence_'+filenames[i]+'.png',255*imgsArray[i][:,:,false_RGB])
            cv.imwrite(cal_extremes_folder+'/sequence_'+filenames[i]+'.png',255*imgsArray[i][:,:,extremes])


if __name__ == '__main__':

    print('Calibration..')
    calibration()
    print('Registration...')
    registration()
    print('Segmentation...')
    segmentation()
