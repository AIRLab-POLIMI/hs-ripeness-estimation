import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi

target_path = '/home/user/data/interim/visualization/calibration_ROIs/'

name = 'home/user/data/raw/2021-09-06/Senop_calibration/Vineyard Cattolica script_220901_203600/Vineyard Cattolica sequence/Vineyard Cattolica sequence_000006/Vineyard Cattolica sequence_000006'
hdr = name+'.hdr'
dat = name+'.dat'
spyFile = envi.open(hdr, dat)
img = np.array(spyFile.load(dtype=np.float32))
false_RGB = [14,8,2]
t = np.zeros((img.shape[0],img.shape[1],3))
for i,b in enumerate(false_RGB) :
    t[:,:,i] = img[:,:,b]
t = t/np.max(t)
t[311:711,311:711,:] = [0.4,0.4,1]

plt.figure()
plt.imshow(t)
plt.savefig(target_path+'zenith_panel.pdf',format='pdf')

ROI =  [[197, 254, 295, 395],[201, 254, 482, 583],[386, 436, 274, 384],[392, 448, 492, 595]]
white = 'home/user/data/raw/2022-08-23/Senop_calibration/Vineyard Cattolica script_240406_121733/Vineyard Cattolica sequence/Vineyard Cattolica sequence_000006/Vineyard Cattolica sequence_000006'
black = 'home/user/data/raw/2022-08-23/Senop_calibration/Vineyard Cattolica script_240406_121503/Vineyard Cattolica sequence/Vineyard Cattolica sequence_000004/Vineyard Cattolica sequence_000004'
hdr = white+'.hdr'
dat = white+'.dat'
spyFile = envi.open(hdr, dat)
img = np.array(spyFile.load(dtype=np.float32))
hdr = black+'.hdr'
dat = black+'.dat'
spyFile = envi.open(hdr, dat)
d_curr = np.array(spyFile.load(dtype=np.float32))
img = img - d_curr

t = np.zeros((img.shape[0],img.shape[1],3))
for i,b in enumerate(false_RGB) :
    t[:,:,i] = img[:,:,b]
t = t/np.max(t)

colors = [[0,0,1],[0,1,0],[1,0,0],[1,1,0]]
for col in range(len(ROI)):
    t[ROI[col][0]:ROI[col][1],ROI[col][2]:ROI[col][3],:] = colors[col]

plt.figure()
plt.imshow(t)
plt.savefig(target_path+'mapir_panel.pdf',format='pdf')