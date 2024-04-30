from .calibration import Calibration
import cv2 as cv
import numpy as np
import glob


class GeometricCalibration(Calibration):
    def __init__(self):
        super().__init__()

    def compute_params(self, cal_folder_vis, cal_folder_nir, int_corner_pattern, selected_range, sensor_boundary):
        """
        Computes the geometric calibration coefficients (i.e. intinsic matrix and
        distortion coefficients, distinct for the two sensors) given a folder
        containing the images divided by band.

                Parameters:
                        cal_folder_vis (str): name of the folder containing the subfolders for the vis_region
                        cal_folder_nir (str): name of the folder containing the subfolders for the nir_region;
                                              if "None", the folder in "cal_folder_vis" is used for every band
                        int_corner_pattern (tuple): number (row,column) of internal corners of the checkerboard
                        selected_range (list): considered band range
                        sensor_boundary (int): separation between bands corresponding to the two sensors (first vis sensor, enumeration strting from 0)
        """

        # lists storing the intrinsic matrices and distortion coefficients for each wavelength
        Mtx = []
        Dist = []

        # sensor_boundary update for consistency, since the "selected_range" is not directly used on the images as in the other cases
        sensor_boundary -= 1

        for k in selected_range:  # for each wavelength
            # select folder
            if (cal_folder_nir == None) or k < sensor_boundary + 1:
                cal_folder = cal_folder_vis
            else:
                cal_folder = cal_folder_nir
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
            objp = np.zeros((int_corner_pattern[0]*int_corner_pattern[1], 3), np.float32)              
            objp[:, :2] = np.mgrid[0:int_corner_pattern[0], 0:int_corner_pattern[1]].T.reshape(-1, 2)
            # arrays to store object points and image points from all the images.
            objpoints = []      # 3d point in real world space
            imgpoints = []      # 2d points in image plane.
            # select the names of the images
            images = glob.glob(cal_folder + '/band_' + str(k+1) + '/*')                 
            for fname in images:  # for all images
                # read the image
                img = cv.imread(fname)
                # convert to grayscale image (if needed)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (int_corner_pattern[0], int_corner_pattern[1]), None)
                # if found, add object points and image points (after refining them)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
                    imgpoints.append(corners2)
            # estimate the intrinsic matrix and distortion coefficients
            _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # collect the results
            Mtx.append(mtx)
            Dist.append(dist)

        # compute sensor specific values by averaging the values of the corresponding wavelengths
        intrinsic_matrix_1 = np.mean(Mtx[sensor_boundary:], axis=0)
        intrinsic_matrix_2 = np.mean(Mtx[:sensor_boundary], axis=0)
        distortion_1 = np.mean(Dist[sensor_boundary:], axis=0)
        distortion_2 = np.mean(Dist[:sensor_boundary], axis=0)

        # assign the parameters
        self.params = {
            "intrinsic_matrix_1": intrinsic_matrix_1,
            "intrinsic_matrix_2": intrinsic_matrix_2,
            "distortion_1": distortion_1,
            "distortion_2": distortion_2
        }

    def load_params(self, distortion_1_name, distortion_2_name, intrinsic_matrix_1_name, intrinsic_matrix_2_name):
        '''
        Loads the geometric calibration coefficients (i.e. intinsic matrix and
        distortion coefficients, distinct for the two sensors) given the files
        (.txt) containing them

            Parameters:
                    intrinsic_matrix_1_name (str): name of the file containing the intrinsic matrix for the first sensor
                    intrinsic_matrix_2_name (str): name of the file containing the intrinsic matrix for the second sensor
                    distortion_1_name (str): name of the file containing the distortion coefficients for the first sensor
                    distortion_2_name (str): name of the file containing the distortion coefficients for the first sensor
        '''

        # Load calibration parameters from file here
        distortion_1 = np.loadtxt(distortion_1_name, delimiter=",")
        distortion_2 = np.loadtxt(distortion_2_name, delimiter=",")
        intrinsic_matrix_1 = np.loadtxt(intrinsic_matrix_1_name, delimiter=",")
        intrinsic_matrix_2 = np.loadtxt(intrinsic_matrix_2_name, delimiter=",")

        self.params = {
            "intrinsic_matrix_1": intrinsic_matrix_1,
            "intrinsic_matrix_2": intrinsic_matrix_2,
            "distortion_1": distortion_1,
            "distortion_2": distortion_2
        }

    def save_params(self, filepath, prefix):
        '''
        Saves the geometric calibration coefficients in a specified folder adding a 
        specified prefix to the names of the files (.txt)

            Parameters:
                    filepath (str): path to the folder in which the files are saved
                    prefix (str): prefix added to the file names

        '''

        np.savetxt(filepath+'/'+prefix+'_intrinsic_matrix_1.txt',self.params["intrinsic_matrix_1"],delimiter=",")
        np.savetxt(filepath+'/'+prefix+'_intrinsic_matrix_2.txt',self.params["intrinsic_matrix_1"],delimiter=",")
        np.savetxt(filepath+'/'+prefix+'_distortion_1.txt',self.params["distortion_1"],delimiter=",")
        np.savetxt(filepath+'/'+prefix+'_distortion_2.txt',self.params["distortion_2"],delimiter=",")

    def apply_calibration(self, selected_range_start, selected_range_stop, sensor_boundary, imgsArray):
        """
        Applies the geometric calibration to a list of images using object parameters

                Parameters:
                        imgsArray (list): list of images on which the geometric calibration is performed
                        selected_range_start (int): first index of the considered band range
                        selected_range_stop (int): last index of the considered band range
                        sensor_boundary (int): index of the separation between bands corresponding to the two sensors (first nir sensor, enumeration strting from 0)
        """
        selected_range = list(range(selected_range_start, selected_range_stop))

        for arr in imgsArray:
            for k in selected_range:        # for each wavelength
                if k < sensor_boundary:     # correct the image based on the calibration coefficients of the corresponding sensor
                    arr[:, :, k] = cv.undistort(arr[:, :, k], self.params["intrinsic_matrix_2"], self.params["distortion_2"])
                else:
                    arr[:, :, k] = cv.undistort(arr[:, :, k], self.params["intrinsic_matrix_1"], self.params["distortion_1"])
