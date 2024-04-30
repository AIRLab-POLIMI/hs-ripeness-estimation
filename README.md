# Grape ripeness estimation via hyperspectral imaging

This repository contains the code to reproduce the experiments reported in the paper titled [On-the-Go Table Grape Ripeness Estimation Via Proximal Snapshot Hyperspectral Imaging](https://dx.doi.org/10.2139/ssrn.4696990)."

# Folders structure

<pre> 
├── hs-vine-data  
│   │   
│   ├── raw  
│   │   ├── bunch_reference: RGB images containing plant and bunches names  
│   │   ├── date    
│   │   │   ├── Senop: raw acqusitions  
│   │   │   └── Senop_calibration: raw acquisitions of radiometric calibration panel    
│   │   ├── lab_analyses: chemical parameters   
│   │   └── senop_geometric_calibration: raw acquisitions of geometric calibration panel  
│   │     
│   ├── interim  
│   │   ├── calibrated: radiometrically and geometrically calibrated data (src/pipeline.py)     
│   │   ├── calibration_panels: plots of the ROI mean spectral signals of the calibration panels    
│   │   ├── geometric_calibration: selected views of the geometric calibration panel stored by spectral band    
│   │   ├── radiometric_plots: radiometric calibration debug plots (src/calibration/radiometric_debug.py)   
│   │   ├── registered: registered data (src/pipeline.py or src/registration/registration.py)   
│   │   │   ├── mutual_information: datasets containing the mutual information metrics  
│   │   │   └── permutation_tests: global and local permutation tests output for comparing registrations    
│   │   ├── segmented   
│   │   │   ├── date    
│   │   │   │   ├── annotations_pure: COCO 1.0 annotations from pre-trained mask R-CNN (src/pipeline.py or src/segmentation/segmentation.py)    
│   │   │   │   ├── annotations_processed: COCO 1.0 annotations elaborated trhough annotation program (CVAT)    
│   │   │   │   └── masked: visualization of the mask R-CNN segmentation (src/segmentation/visualization.py)      
│   │   │   └── date-hypercubes: visualization of the annotations after (CVAT) processing (src/segmentation/build_dataset.py)    
│   │   ├── geometric_calibration: acquisitions of geometric calibration panel divided by sensor and band   
│   │   └── visualization: additional plots     
│   │    
│   ├── processed   
│   │   └── date    
│   │       ├── dataset.csv: final dataset in .csv (src/segmentation/build_dataset.py)   
│   │       ├── dataset.xlsx: final dataset in .xlsx (src/segmentation/build_dataset.py)     
│   │       └── spectral plots: batch plots of spectral signals (src/segmentation/build_dataset.py)     
│   │   
│   └── results: prediction results (src/prediction/pls.py) 
|       
├── hs-ripeness-estimation      
│   │    
│   ├── conf    
│   │   ├── config.yaml: default configurations     
│   │   └── db      
│   │       └── conf_date.yaml: date-specific configurations        
|   |       
│   ├── models   
│   │   ├── calibration: intrinsic matrices and distortion coefficients for geometric calibration   
│   │   └── segmentation: weights of the pre-trained mask R-CNN     
|   |       
│   └── src     
│       ├── calibration      
│       │   ├── calibration.py: abstract class for calibration classes   
│       │   ├── radiometric_calibration.py: class for radiometric calibration   
│       │   ├── radiometric_debug.py: radiometric calibration plots     
│       │   ├── geometric_calibration.py: class for geometric calibration   
│       │   ├── panel_ROI.py: manual annotation of ROIs on the calibration panel    
│       │   └── matlab: matlab files    
│       ├── registration    
│       │   ├── registration.py: applies registration   
│       │   ├── mutual_information.py: computes the mutual information metric for each band for each image  
│       │   └── permutation_tests.py: applies global and local permutation tests based on the mutual information    
│       ├── segmentation    
│       │   ├── configs: configurations of pre-trained mask R-CNN   
│       │   ├── visualization.py: visualizes the predictions of the pre-trained mask R-CNN      
│       │   ├── segmentation.py: applies the pre-trained mask R-CNN     
│       │   └── build_dataset.py: builds the final dataset given the processed COCO 1.0 annotations     
│       ├── prediction      
│       │   └── pls.py: performs PLSR regression with nested CV and single loop CV evaluation
│       ├── visualization: additional plots        
│       └── pipeline.py: applies the calibration, registration and pre-trained segmentation pipeline    
|   
├── README.md       
└── .gitignore 
</pre> 

# Example of hyperspectral processing (2021/09/06):

$ python pipeline.py &nbsp;&nbsp;\\     
&nbsp;&nbsp;&nbsp;&nbsp;db=conf_2021_09_06

% the segmented annotations are stored in interim/segmented/2021-09-06/annotations_pure     
% after elaboration in CVAT, put the annotations in interim/segmented/2021-09-06/annotations_processed

$ python segmented/build_dataset.py &nbsp;&nbsp;\\  
&nbsp;&nbsp;&nbsp;&nbsp;db=conf_2021_09_06

% The resulting dataset is stored in processed/2021-09-06


# Example of prediction (bunches, TSS):

$ python prediction/pls.py &nbsp;&nbsp;\\   
&nbsp;&nbsp;&nbsp;&nbsp;db.prediction.analysis=bunches &nbsp;&nbsp;\\   
&nbsp;&nbsp;&nbsp;&nbsp;db.prediction.chemical_target=Brix  


# Example of prediction (plants, anthocyanins):

$ python prediction/pls.py &nbsp;&nbsp;\\   
&nbsp;&nbsp;&nbsp;&nbsp;db.prediction.analysis=plants &nbsp;&nbsp;\\    
&nbsp;&nbsp;&nbsp;&nbsp;db.prediction.chemical_target=Antociani
