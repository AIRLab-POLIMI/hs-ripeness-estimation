'''
Grape instance segmentation visualization on RGB vine images through a pretrained mask RCNN architecture
Functions:
- inference_and_plots: applies and visualizes the instance segmentation
Main: 
- segmentation: application of the segmentation
'''

# libraries
#import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2 as cv
import os
import glob
from tqdm import tqdm
import re
import hydra
from omegaconf import DictConfig


def inference_and_plots(images, names, cfg, predictor, output_folder, apply_mask, ext='.png', scale=1.):
    '''
    Applies the mask RCNN, visualizing and saving masks and bounding boxes

        Parameters:
                images (list): list of BGR to be segmented
                names (list): list of names for saving the visualization for each image
                cfg: architecture weights and configurations
                predictor: detectron2 predictor
                output_folder (str): folder in which the visualizations are saved
                apply_mask (bool): if true, single object masks and visualizations are saved
                ext (str): extension of the saved images
                scale (float): scale of the visualizations

        Returns:
                Boxes (list): list of predicted bounding boxes
                Classes (list): list of predicted classes
                Masks (list): list of predicted masks
    '''

    # lists for storing the outputs
    Boxes = []
    Classes = []
    Masks = []

    # for each image
    for i in tqdm(range(len(images))):

        # make the predictions and visualize them on the original image
        outputs = predictor(images[i])
        v = Visualizer(images[i][:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=scale)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # save the visualization
        cv.imwrite(output_folder+'/'+names[i]+ext, out.get_image()[:, :, ::-1])

        # predictions for the image
        B = outputs["instances"].pred_boxes
        C = outputs["instances"].pred_classes
        M = outputs["instances"].pred_masks

        # if we want to save the masked instances 
        if apply_mask:
            
            # for each instance in the image, save an image of the instance
            for ind, mask in enumerate(M):
                mask = mask.cpu().numpy().reshape((images[i].shape[0], images[i].shape[1], 1))
                masked = images[i]*mask
                cv.imwrite(output_folder+'/masked/'+names[i]+'_'+str(ind)+ext, masked)
                # np.savetxt(output_folder+'/masked/'+names[i]+'_'+str(ind)+'_m'+'.txt',mask)

        # append the predictions of the image
        Boxes.append(B)
        Classes.append(C)
        Masks.append(M)

    return Boxes, Classes, Masks


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def segmentation(cfg: DictConfig):
    '''
    Application of the visualization. Saves the segmented RGB images. If "apply_mask" = True,
    also saves an image for each instance in which the mask is applied to the RGB image.
    '''

    config_file = cfg.db.segmentation.config_file
    weights_file = cfg.db.segmentation.weights_file
    source_folder = cfg.db.segmentation.source_folder
    RGB_prefix = cfg.db.segmentation.RGB_prefix
    RGB_folder = source_folder + RGB_prefix
    output_folder = cfg.db.segmentation.target_folder
    threshold = cfg.db.segmentation.threshold
    scale = cfg.db.segmentation.scale
    apply_mask = cfg.db.segmentation.apply_mask

    # data location
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if apply_mask:
        if not os.path.exists(output_folder+'/masked'):
            os.makedirs(output_folder+'/masked')

    # names of the images
    im_files = glob.glob(RGB_folder + '/*')

    # lists for storing the images and their names
    images = []
    names = []

    # for each image
    for im_path in im_files:

        # read it and append its information to the predisponed lists
        img = cv.imread(im_path)  # OpenCV loads images in BGR format by default
        full_name = im_path[:-4]
        name = re.split('/',full_name)[-1]
        images.append(img)
        names.append(name)
 
    # setup the predictor
    det2_cfg = get_cfg()
    det2_cfg.merge_from_file(config_file)
    det2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    det2_cfg.MODEL.WEIGHTS = weights_file
    predictor = DefaultPredictor(det2_cfg)

    # make predictions and save the visualization
    Boxes, Classes, Masks = inference_and_plots(images=images, names=names, cfg=det2_cfg, predictor=predictor, output_folder=output_folder, apply_mask=apply_mask, ext='.png', scale=scale)


if __name__ == "__main__":

    segmentation()
