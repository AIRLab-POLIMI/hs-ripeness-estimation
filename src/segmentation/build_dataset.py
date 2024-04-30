'''
Given the chemical dataset and a corresponding .json file containing COCO 1.0 annotations that is linked
to the dataset, creates a dataset in which, for each instance of annotated grapes, the mean spectrum spectral
signal over the selected wavelengths is computed and associated with the corresponding chemical measurements:
- get_annotated_data: recovers annotations from .json file
Main: 
- apply_seg: builds dataset from annotations and file containing the chemical recordings
'''

# libraries
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import json
from imantics import Polygons
import spectral.io.envi as envi
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import iqr



def get_annotated_data(filename,img_dir):
    '''
    Given a directory containing the .json annotation file, returns a list of dictionaries containing information 
    about the single instances.

        Parameters:
            filename (str): path to the file containing the annotations
    
        Returns:
            annotations (list(dict)): list of dictionaries of the annotations
    '''

    # open and read the annotation file
    f = open(filename)
    data = json.load(f)
    image_info = data["images"]              # list of dicts
    annotations_info = data["annotations"]   # list of dicts 

    # create dict that associates id with file_name
    images = {}
    for im_inf in image_info:
        images[im_inf["id"]] = im_inf["file_name"][:-4]

    # list for storing relevant segmentations and relative filenames and info
    annotations = []

    # recover information about relevant annotations and append them to the list
    for an_inf in annotations_info:
        if an_inf["attributes"]["tagged"]:       
            record = {}
            record["file_name"] = images[an_inf["image_id"]]
            record["plant_no"] = an_inf["attributes"]["plant_no"]
            record["grape_no"] = an_inf["attributes"]["grape_no"]
            seg = an_inf["segmentation"]
            # transform the polygon annotation into binary mask
            img = cv.imread(img_dir+'/'+images[an_inf["image_id"]]+'.png')
            height = img.shape[0]
            width = img.shape[1]
            mask = Polygons(seg).mask(width = width, height = height)
            record["segmentation"] = mask.array.reshape(height,width,1)

            annotations.append(record)

    f.close()

    return annotations


def IQROutliers(masked_img, mask, selected_range, debug, false_RGB, new_name, cleaning_folder, tolerance=2.2):
    '''
    Given a masked image, performs interquartile range (IQR) outlier datection and returns the cleaned mask from
    which the pixels flagged as outliers are removed

        Parameters:
            masked_img (np.array): original masked image
            mask (np.array): mask used to obtain the masked image
            selected_range (list): list of indexes of the bands to be considered
            debug (bool): if True, creates a debug folder in which false RGB images show the orginal masked image with
                          the outlier pixels colored in red
            false_RGB (list): indexes of the bands used to simulate the false RGB images in the debug folder
            new_name (str): name of the fase RGB images of the debug folder, used for consistency w.r.t. the dataset annnotations
            cleaning_folder (str): path to the debug folder
            tolerance (float): tolerance factor of the IQR method for flagging the outliers
    
        Returns:
            new_img (np.array): cleaned masked image, in which the outlier pixels are set to 0 
    '''

    # obtain the masked (non-zero) entries of image
    entries = np.nonzero(mask.reshape(mask.shape[0],mask.shape[1]))
    data = np.array(masked_img[entries])
    data = data[:,selected_range]

    # list containing the location of the outlying pixels for each band
    out = []
    # for each band
    for c in range(data.shape[1]):
        # compute the IQR andthe median
        iqr_ = iqr(data[:,c])
        median = np.median(data[:,c])
        # flag as outliers all pixels that are further from the median than the IQR multiplied by the tolerance factor
        lower = median - tolerance*iqr_
        upper = median + tolerance*iqr_
        outliers = ((data[:,c]>upper) | (data[:,c]<lower))
        # append the band-specific outliers
        out.append(outliers)

    # flag as outliers all pixels that are flagged as outliers for at least one band
    out_mask = np.zeros(data.shape[0],bool)
    for i in range(len(out)):
        out_mask = np.logical_or(out_mask,out[i])

    # set the value of the outlier pixels to 0
    new_entries = (entries[0][out_mask],entries[1][out_mask])
    new_img = np.array(masked_img)
    for i in range(len(new_entries[0])):
        new_img[new_entries[0][i],new_entries[1][i],:] = 0

    # if debug==True, save the corresponding debug image
    if debug:
        new_folder = cleaning_folder
        new_img_debug = np.array(masked_img)
        for i in range(len(new_entries[0])):
            new_img_debug[new_entries[0][i],new_entries[1][i],false_RGB] = [0,0,1]

        cv.imwrite(new_folder+'/'+new_name+'.png',255*new_img_debug[:,:,false_RGB])

    # return the cleaned masked image
    return new_img


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def apply_seg(cfg: DictConfig):
    '''
    Construction of the dataset from annotations and chemicals files. Optionally, plots and files for the segmented
    hypercubes, RGB and "extremes" images and mean spectral signals can be provided
    '''

    annotations_file = cfg.db.segmentation.annotation_CVAT + '/instances_default.json'
    source_folder = cfg.db.segmentation.source_folder
    output_folder = cfg.db.segmentation.target_folder + '-hypercubes'
    plot_flag= cfg.db.segmentation.hypercubes_plots
    save_hypercubes = cfg.db.segmentation.hypercubes_files
    false_RGB_folder = output_folder+'/false_RGB'
    extremes_folder = output_folder+'/extremes'
    false_RGB = cfg.db.segmentation.false_RGB
    selected_range_start = cfg.db.segmentation.selected_range_start
    selected_range_stop = cfg.db.segmentation.selected_range_stop
    selected_range = list(range(selected_range_start,selected_range_stop))
    extremes = [selected_range[-1],selected_range[len(selected_range)//2],selected_range[0]]
    chemical_file = cfg.db.segmentation.chemical_file
    sheet_name = cfg.db.segmentation.sheet_name
    year_bands = cfg.db.segmentation.year_bands
    chem_ind = cfg.db.segmentation.chem_ind
    new_df_folder = cfg.db.segmentation.new_df_folder
    new_df_name = cfg.db.segmentation.new_df_name
    if sheet_name == 'Vendemmia':
        new_df_name = new_df_folder + '/' + new_df_name
    else:
        new_df_name = new_df_folder + '/plants_' + new_df_name
    excel_copy = cfg.db.segmentation.excel_copy
    plot_spectra = cfg.db.segmentation.plot_spectra
    plot_batch_size = cfg.db.segmentation.plot_batch_size
    cleaning_debug = cfg.db.segmentation.cleaning_debug
    cleaning_plots_folder = output_folder+'/cleaning'
    cleaning_parameter = cfg.db.segmentation.cleaning_parameter

    # data location
    if save_hypercubes:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    if plot_flag:
        if not os.path.exists(false_RGB_folder):
            os.makedirs(false_RGB_folder)
        if not os.path.exists(extremes_folder):
            os.makedirs(extremes_folder)
    if not os.path.exists(new_df_folder):
        os.makedirs(new_df_folder)
    if cleaning_debug:
        if not os.path.exists(cleaning_plots_folder):
            os.makedirs(cleaning_plots_folder)
    
    # retrieve annotations
    annotations = get_annotated_data(annotations_file,source_folder+'/false_RGB')

    # retrieve chemical measurements
    df = pd.read_excel(chemical_file, sheet_name = sheet_name)

    # list for storing data of the new dataset
    data_for_df = []
    plants = []

    # data structures used for storing the results
    plants_dict = {}

    # for each annotation
    for ann in tqdm(annotations):

        # recover the plant number
        plant = str(int(ann["plant_no"]))
        # check on the presence of the datum in the chemical dataset
        if sheet_name != 'Vendemmia':
            if np.sum(np.logical_and(df['Data'] == np.datetime64(cfg.db.date_str),df['Pianta'] == int(plant))) == 0:
                continue


        # recover corresponding hypercube
        file_name = ann["file_name"]
        full_name = source_folder+'/'+file_name
        spyFile = envi.open(full_name+'.hdr',full_name+'.dat')
        img = np.array(spyFile.load(dtype=np.float32))

        # apply the segmentation mask to the hypercube
        mask = ann["segmentation"]
        masked_img = img*mask

        # instance name: "gipj", where "gi" refers to the bunch number and "pj" refers to the plant
        new_name = 'g'+str(int(ann["grape_no"]))+'p'+str(int(ann["plant_no"]))
        plants.append(int(ann["plant_no"]))

        # check on the presence of the datum in the chemical dataset
        if sheet_name == 'vendemmia':
            if np.sum(df['Codice grappolo'] == new_name) == 0:
                continue
        
        # clean the masked image by IQR outlier detection
        masked_img = IQROutliers(masked_img=masked_img, mask=mask, selected_range=selected_range, debug=cleaning_debug, false_RGB=false_RGB, new_name=new_name, cleaning_folder=cleaning_plots_folder, tolerance=cleaning_parameter)

        # optional plots
        if save_hypercubes:
            envi.save_image(hdr_file=output_folder+'/'+new_name+'.hdr', image=masked_img, ext='.dat',interleave='bsq')
        if plot_flag:
            cv.imwrite(false_RGB_folder+'/'+new_name+'.png' ,255*masked_img[:,:,false_RGB])
            cv.imwrite(extremes_folder+'/'+new_name+'.png' ,255*masked_img[:,:,extremes])

        # list for storing information of the single datum
        datum = []

        # add name
        if sheet_name == 'Vendemmia':
            datum.append(new_name)

        # additional data structure to contain the sum of the reflectance values at each wavelength;
        # the use of curr_sum and curr_n_elem is used for the construction of the plants dataset: we
        # scan through all bunches and save the sum of the reflectance values at each wavelength and
        # the number of elements in the masked image. Once the scan is finished, we obtain the mean
        # reflected value of the plant by dividing the sum of all reflectance values of the bunches that
        # are part of the plant by the total sum of pixels in the images of those bunches
        curr_sum = []

        # compute the mean (or summed) spectrum for each selected wavelength
        for k in selected_range:
            curr = masked_img[:,:,k]
            m = np.nonzero(curr)
            # consistency check on the existance of non-zero entries in the masked image due to possible mis-registraions
            if np.size(m):
                final = np.mean(curr[m])
                curr_sum.append(np.sum(curr[m]))
                if k == selected_range[0]:
                    curr_n_elem = len(curr[m])
            else:
                final = 0
                curr_sum.append(0)
                if k == selected_range[0]:
                    curr_n_elem = 0
            if sheet_name == 'Vendemmia':
                datum.append(final)
        curr_sum = np.array(curr_sum)

        # recover the chemical information
        # if we have chemical information for each distinct bunch
        if sheet_name == 'Vendemmia':
            chem = df[df['Codice grappolo'] == new_name].values[0][chem_ind:]
        # else use mean plant value
        else:
            # if a bunch of the same plants has already been encountered, add the new spectral information
            if plant in plants_dict:
                plants_dict[plant]['n_elements'] = plants_dict[plant]['n_elements'] + curr_n_elem
                plants_dict[plant]['band_sums'] = plants_dict[plant]['band_sums'] + curr_sum
            # else recover the chemical information and initialize the spectral information
            else:                    
                chem = df[np.logical_and(df['Data'] == np.datetime64(cfg.db.date_str),df['Pianta'] == int(plant))].values[0][cfg.db.segmentation.curve_start_ind:cfg.db.segmentation.curve_end_ind]
                plants_dict[plant] = {
                    'n_elements':curr_n_elem,
                    'band_sums':curr_sum,
                    'chems':chem
                }
        
        # add the chemical information to the datum
        if sheet_name == 'Vendemmia':
            for e in chem:
                datum.append(e)
            data_for_df.append(datum)

    # create colnames for new dataset
    colnames = []
    colnames.append('id')
    for k in selected_range:
        colnames.append(str(year_bands[k]))
    if sheet_name == 'Vendemmia':
        for i in range(len(df.columns[chem_ind:])):
            colnames.append(df.columns[chem_ind:][i])
    else:
        for i in range(len(df.columns[cfg.db.segmentation.curve_start_ind:cfg.db.segmentation.curve_end_ind])):
            colnames.append(df.columns[cfg.db.segmentation.curve_start_ind:cfg.db.segmentation.curve_end_ind][i])

    # if a plant analysis is performed, since the scan is finished, recover the mean reflectance values
    if sheet_name != 'Vendemmia':

        for p in plants_dict:
            datum = []
            curr_dict = plants_dict[p]
            band_sums = curr_dict['band_sums']
            n_elem = curr_dict['n_elements']
            chems = curr_dict['chems']
            band_sums = band_sums/n_elem

            # append the datum
            datum.append('p'+p)
            for s in band_sums:
                datum.append(s)
            for e in chems:
                datum.append(e)
            data_for_df.append(datum)



    # assemble and save the new dataset
    df_new = pd.DataFrame(data = data_for_df, columns = colnames)
    df_new.to_csv(new_df_name+'.csv', index=False)
    if excel_copy:
        df_new.to_excel(new_df_name+'.xlsx', index=False)


    # optional plots
    if plot_spectra:
        
        # for naming issues with few data
        i = -1

        spectra_folder = new_df_folder + '/spectral_plots'
        if not os.path.exists(spectra_folder):
            os.makedirs(spectra_folder)

        data = np.array(df_new.iloc[:,1:(len(selected_range)+1)])

        columns = [str(year_bands[e]) for e in selected_range]

        # batch plots of single spectra
        for i in range(data.shape[0]//plot_batch_size):
            plt.figure()
            leg = []
            for j in range(plot_batch_size):
                plt.plot(selected_range, data[i*plot_batch_size+j,:])
                leg.append(df_new.id[i*plot_batch_size+j])
                plt.xticks(selected_range[::5],columns[::5])
                plt.xlabel('Wavelengths')
                plt.ylabel('Reflectance')
                plt.legend(leg)
            plt.savefig(spectra_folder+'/'+'batch_'+str(i)+'.png',bbox_inches='tight')
        if data.shape[0]%plot_batch_size != 0:
            plt.figure()
            leg = []
            for j in range(data.shape[0]%plot_batch_size):
                plt.plot(selected_range, data[(-1 -j),:])
                leg.append(df_new.id[len(df_new.id) -1 -j])
                plt.xticks(selected_range[::5],columns[::5])
                plt.xlabel('Wavelengths')
                plt.ylabel('Reflectance')
                plt.legend(leg)
            plt.savefig(spectra_folder+'/'+'batch_'+str(i+1)+'.png',bbox_inches='tight')

        # plot of the spectra grouped by row
        plt.figure()
        for j in range(data.shape[0]):
            if sheet_name == 'Vendemmia':
                color = 'red' if plants[j] < 14 else 'blue'
            else:
                color = 'red' if int(df_new.id.values[j][1:]) < 14 else 'blue'
            plt.plot(selected_range, data[j,:],color)
            plt.xticks(selected_range[::5],columns[::5])
            plt.xlabel('Wavelengths')
            plt.ylabel('Reflectance')
        plt.savefig(spectra_folder+'/row_division.png')

        # plot of spectra grouped by plants
        if sheet_name == 'Vendemmia':
            colors = ['black','dimgray','lightgray','rosybrown','lightcoral',
            'brown','red','sienna','chocolate','tan',
            'orange','gold','olive','yellow','yellowgreen',
            'darkolivegreen','lime','aquamarine','teal','cyan',
            'deepskyblue','steelblue','navy','mediumslateblue','magenta',
            'deeppink']
            plt.figure()
            for j in range(data.shape[0]):
                plt.plot(selected_range, data[j,:],color=colors[plants[j]])
                plt.xticks(selected_range[::5],columns[::5])
                plt.xlabel('Wavelengths')
                plt.ylabel('Reflectance')
            plt.savefig(spectra_folder+'/plant_division.png')


if __name__ == '__main__':

    apply_seg()