'''
Computation of the Mutual Intensity metric of all bands w.r.t. the reference image.
'''

# libraries
import numpy as np
import spectral.io.envi as envi
import glob
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pandas as pd
import SimpleITK as sitk
from skimage.metrics import normalized_mutual_information
import re

@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def mutual_info(cfg: DictConfig):

    date = str(cfg.db.date)
    target_folder = '/home/user/data/interim/registered/mutual_information'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    selected_range_start = cfg.db.registration.general.selected_range_start
    selected_range_stop = cfg.db.registration.general.selected_range_stop
    selected_range = list(range(selected_range_start,selected_range_stop))
    year_bands = cfg.db.calibration.radiometric.year_bands
    reference_band = cfg.db.registration.intensity.ref_band
    
    registrations = glob.glob('/home/user/data/interim/registered/'+date+'*')
    for reg in registrations:
        df_name = re.split('/',reg)[-1]
        data = []
        print(df_name)
        # recover the files inside the data folder
        names = []
        for name in sorted(glob.glob(reg+'/*.hdr')):
            names.append(name[:-4])

        # for each hyperspectral image, compute the MI
        for name in tqdm(names):
            mi = []
            spyFile = envi.open(name+'.hdr',name+'.dat')
            img = np.array(spyFile.load(dtype=np.float32))
            for w in selected_range:
                if w != reference_band:
                    mi.append(normalized_mutual_information(img[:,:,reference_band],img[:,:,w],bins=100))
            data.append(mi)

        # save the MI dataset in the target folder
        columns = []      
        for w in selected_range:
            if w != reference_band:
                columns.append(str(year_bands[w]))
        df = pd.DataFrame(data=data, columns=columns)
        df.to_excel(target_folder+'/'+df_name+'.xlsx', index=False)

if __name__ == '__main__':
    mutual_info()