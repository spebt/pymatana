import sys
import math
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import rich.progress
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import json

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)

# Matrix dimension 90*90*32*6*8
with open('../SysMatConfig/Parameters.json') as json_file:
    parameters = json.load(json_file)

NImgX_ = parameters["numImageX"]
NImgY_ = parameters["numImageY"]
NDetY_ = parameters["pixelSiPM"]
NModule_ = parameters["numPanel"]
NDetX_ = parameters["numDetectorLayer"]

# #
# bg = 0.2
# radiusR = 20
# bgRadius = radiusR*2
# rDot = np.linspace(2, 4.5, 6)*radiusR/20
# #
addNoise = parameters["AddPoisson"]
radiusR = 20
bgRadius = radiusR*2
rDot = np.linspace(2, 4.5, 6)*radiusR/20
phantom_activity = 1e9

if addNoise:   
    contrasts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    voxel_bkg = phantom_activity/(3.14*((NImgX_*NImgY_) + (rDot[0]*rDot[0])*4 + (rDot[1]*rDot[1])*4 + (rDot[2]*rDot[2])*4 
                +(rDot[3]*rDot[3])*4 +(rDot[4]*rDot[4])*4 +(rDot[5]*rDot[5])*4 ))
    bg = voxel_bkg
else:
    contrasts = range(1, 7)
    voxel_bkg = 1
    bg = 0.2


# values = range(1, 7)
NIteration = parameters["ReconstructionIterations"]

scale = 100
# Read in the reconstructed data
inFname = 'images/contrast-recon-data.npz'
dataUnpack = np.load(inFname)
# dataSize = 50
reconData = dataUnpack['arr_0']
print(f"Reconstructed data size: {reconData.shape}")
Nframes=reconData.shape[0]
phantom=np.load('../ImageReconstructor/input/circle-phantom.npz')['arr_0']

# Plot the CNR
# plt.rcParams["figure.figsize"] = (16, 12)

# Calculate the CNR
bgIndex = np.nonzero(phantom.flatten() == bg)
objIndex = []
for idx in range(0, 6):
    objIndex.append(np.nonzero(phantom.flatten() == contrasts[idx]*voxel_bkg))
# print(len(bgIndex[0]))
# print(reconData[0][objIndex[0]].shape)
CNRs = np.zeros((6, Nframes))
bgMeans=np.zeros(Nframes)
bgStds=np.zeros(Nframes)
print(reconData.shape)
scale = 100
endIdx = int(parameters["ReconstructionIterations"]/scale)
for iFrame in range(0, endIdx):
      bgMean = np.mean(reconData[iFrame][bgIndex])
      bgStd = np.std(reconData[iFrame][bgIndex])
      bgMeans[iFrame] = bgMean
      bgStds[iFrame] = bgStd
      for idx in range(0, 6):
        objMean=np.mean(reconData[iFrame][objIndex[idx]])
        CNRs[idx, iFrame]=(objMean - bgMean)/bgStd

fig, axs = plt.subplots(3, 1, figsize=(8, 6), dpi=150,sharex=True)
axs[0].set_title("Background Mean")
axs[0].plot(np.arange(1,NIteration,100),bgMeans)
axs[0].set_xlim(-1,NIteration)
axs[1].set_title("Background Standard Deviation")
axs[1].plot(np.arange(1,NIteration,100),bgStds)
axs[2].set_title("Background Mean/Std")
axs[2].plot(np.arange(1,NIteration,100),bgMeans/bgStds)
axs[2].set_xlabel("Number of Iterations")
plt.tight_layout()

fig, axs = plt.subplots(6, 1, figsize=(8, 6), dpi=150,sharex=True)
axs[0].set_xlim(-1,NIteration)
for idx in range(0,6):
    axs[idx].plot(np.arange(1,NIteration,100), CNRs[idx])
    axs[idx].annotate(f'Circle # {idx}', xy=(0,axs[idx].get_ylim()[1]))
axs[5].set_xlabel("Number of Iterations")
plt.tight_layout()
plt.savefig('images/CNR_Image.png')
plt.show()