'''
Grape instance segmentation on RGB vine images through a pretrained mask RCNN architecture
Functions:
- get_dicts: loads RGB images folder as list of dicts for detectron2 inference
- return_polygons: sets the COCO annotations file into a more favorable format for CVAT elaboration
Main: 
- segmentation: applies the instance segmentation and retuns a corresponding .json file in COCO 1.0 annotation format
'''

# libraries
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import os
import glob
import re
import hydra
from omegaconf import DictConfig
import json
from detectron2.data import DatasetCatalog
import pycocotools.mask as mask_util
from imantics import Mask  
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import cv2 as cv
import shutil

def get_dicts(img_dir):
    '''
    Given a directory containing the images, returns a list of dictionaries containing information about the images.
    Used for passing and registering the images as a detectron2 dataset.

        Parameters:
            img_dir (str): path to the folder containing the images
    
        Returns:
            dataset_dicts (list(dict)): list of dictionaries of the images
    '''
    
    dataset_dicts = []
    for idx, v in enumerate(sorted(glob.glob(img_dir+'/*'))):
        record = {}
        
        filename = v
        img = cv.imread(filename)
        height = img.shape[0]
        width = img.shape[1]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["annotations"] = []

        dataset_dicts.append(record)

    return dataset_dicts


def return_polygons(json_RLE, img_dir, output_dir):
    ''''
    Modifies the COCO annotations file returned from detectron2. It adds informations and converts the 
    segmentation results from compressed RLE to polygon format

        Parameters:
                json_RLE (str): path to the original .json annotation file
                img_dir (str): path to the RGB image folder, used for recovering correct notation consistently
                output_dir (str): path to the output folder in which the final .json file is saved
    '''

    # recover the image names. Consistency granted by lexicographic sorting of the names
    names = []
    for name in sorted(glob.glob(img_dir+'/*.png')):
        names.append(re.split('/',name)[-1])

    # load .json annotation file returned by detectron2
    json_file = os.path.join(json_RLE)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # initialization of final dictionary
    final_dict = {}

    # initialization for correct COCO 1.0 format
    final_dict["licenses"] = {
        "name": "",
        "id": 0,
        "url": ""
    }
    final_dict["info"] = {
        "contributor": "Manuel Piliego",
        "date_created": "",
        "description": "Instance segmentation dataset of Red Globe grape bunches. Images have been captured in summer 2021.",
        "url": "",
        "version": "",
        "year": "2023"
    }
    final_dict["categories"] = [{
        "id": 1,
        "name": "bunch",
        "supercategory": ""
    }]

    # list of dictionaries containing image information
    images_dicts = []
    for i in range(len(names)):
        im_record = {}
        im_record["id"] = i+1
        img = cv.imread(img_dir+'/'+names[i])
        height = img.shape[0]
        width = img.shape[1]
        im_record["width"] = width
        im_record["height"] = height
        im_record["file_name"] = names[i]
        im_record["license"] = 0
        im_record["flickr_url"] = ""
        im_record["coco_url"] = ""
        im_record["date_captured"] = 0
        images_dicts.append(im_record)

    final_dict["images"] = images_dicts

    # list of dictionaries containing annotation information 
    annotations_dicts = []
    id_in_image = 1
    prev_name = imgs_anns[0]["image_id"]
    for v in imgs_anns:
        record = {}

        if v["image_id"] != prev_name:
            id_in_image = 1
        prev_name = v["image_id"]
        record["image_id"] = v["image_id"]+1
        record["id"] = id_in_image
        id_in_image += 1
        record["category_id"] = 1
        record["bbox"] = v["bbox"]
        record["score"] = v["score"]
        # revover compressed RLE segmentation
        segment = {
            "counts": v["segmentation"]["counts"],
            "size": v["segmentation"]["size"]
        }
        # transform RLE in polygon
        mask = mask_util.decode(segment)[:, :]
        polygons = Mask(mask).polygons()
        record["segmentation"] = []
        for l in polygons.segmentation:
            if len(l) > 5:
                record["segmentation"].append(l)
        record["iscrowd"] = 0

        annotations_dicts.append(record)
    final_dict["annotations"] = annotations_dicts
    
    # save new .json file containing COCO 1.0 annotations for CVAT elaboration
    json_object = json.dumps(final_dict, indent = 2)
    jsonFile = open(output_dir+'/annotated_data.json', "w")
    jsonFile.write(json_object)
    jsonFile.close()


@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def segmentation(cfg: DictConfig):
    '''
    Application of the segmentation. The predictions are saved in COCO 1.0 .json format; in particular,
    the segmentations are reported as polygons.
    '''

    config_file = cfg.db.segmentation.config_file
    weights_file = cfg.db.segmentation.weights_file
    source_folder = cfg.db.segmentation.source_folder
    RGB_prefix = cfg.db.segmentation.RGB_prefix
    RGB_folder = source_folder + RGB_prefix
    threshold = cfg.db.segmentation.threshold
    data_path = cfg.db.segmentation.data_path
    annotation_path = cfg.db.segmentation.annotation_path
    annotation_CVAT = cfg.db.segmentation.annotation_CVAT
    data_dir = RGB_folder

    # data location
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    # we also create a directory for storing the annotation file after CVAT processing
    if not os.path.exists(annotation_CVAT):
        os.makedirs(annotation_CVAT)

    # setup the predictor
    det2_cfg = get_cfg()
    det2_cfg.merge_from_file(config_file)
    det2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    det2_cfg.MODEL.WEIGHTS = weights_file
    predictor = DefaultPredictor(det2_cfg)
 
    # register the dataset
    DatasetCatalog.register("my_data", lambda: get_dicts(RGB_folder))
    MetadataCatalog.get("my_data").set(thing_classes=["bunch"])

    # inference through detectron2. Saves initial .json file with annotations
    evaluator = COCOEvaluator('my_data', det2_cfg, False, output_dir="/workspace/src/segmentation/output/")
    val_loader = build_detection_test_loader(det2_cfg, 'my_data')
    inference_on_dataset(predictor.model, val_loader, evaluator)

    # convert .json annotation file to correct format
    return_polygons(data_path, data_dir, annotation_path)

    shutil.rmtree('/workspace/src/segmentation/output')

    



if __name__ == "__main__":

    segmentation()

    
    

