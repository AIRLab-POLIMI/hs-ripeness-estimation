import cv2
import numpy as np
import sys
import spectral.io.envi as envi

class BoundingBoxWidget(object):
    def __init__(self,name):
        self.original_image = cv2.imread(name)
        self.clone = self.original_image.copy()

        cv2.namedWindow('Calibration ROI(s)',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration ROI(s)', 750,750)
        cv2.setMouseCallback('Calibration ROI(s)', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []
        self.final_coordinates = np.empty((0,4),int)

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            if self.image_coordinates[0][1] < self.image_coordinates[1][1]:
                start_row = self.image_coordinates[0][1]
                end_row = self.image_coordinates[1][1]
            else: 
                start_row = self.image_coordinates[1][1]
                end_row = self.image_coordinates[0][1]
            if self.image_coordinates[0][0] < self.image_coordinates[1][0]:
                start_col = self.image_coordinates[0][0]
                end_col = self.image_coordinates[1][0]
            else:
                start_col = self.image_coordinates[1][0]
                end_col = self.image_coordinates[0][0]
            self.final_coordinates = np.append(self.final_coordinates,np.array([[start_row,end_row,start_col,end_col]]), axis = 0)
            # Draw rectangle 
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("Calibration ROI(s)", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.final_coordinates = np.empty((0,4),int)

    def show_image(self):
        return self.clone

if __name__ == '__main__':

    if len(sys.argv)>1:
        final_name = sys.argv[1]
    else:
        final_name = 'test'

    hdr = 'white_reference_2022\Vineyard Cattolica sequence_000006.hdr'
    dat = 'white_reference_2022\Vineyard Cattolica sequence_000006.dat'
    spyFile = envi.open(hdr, dat)
    hypercube = np.array(spyFile.load(dtype=np.float32))

    false_RGB = [14,8,1] # [1,8,14]
    img = hypercube[:,:,false_RGB]/np.max(hypercube[:,:,false_RGB])*255

    cv2.imwrite(final_name+'_false_RGB.png',img)

    boundingbox_widget = BoundingBoxWidget(final_name+'_false_RGB.png')
    go_on = True
    while go_on:
        cv2.imshow('Calibration ROI(s)', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            go_on = False

    print('ROI(s)')
    print(boundingbox_widget.final_coordinates)

    img_ROI = boundingbox_widget.original_image
    for i in range(boundingbox_widget.final_coordinates.shape[0]):
        img_ROI[boundingbox_widget.final_coordinates[i,0]:boundingbox_widget.final_coordinates[i,1],boundingbox_widget.final_coordinates[i,2]:boundingbox_widget.final_coordinates[i,3],:] = [0,0,255]
    cv2.imwrite(final_name+'_ROI.png',img_ROI)

    np.savetxt(final_name+'_ROI.txt',boundingbox_widget.final_coordinates)

