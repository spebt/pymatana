import numpy as np
import matplotlib.pyplot as plt
import json
import struct
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import rich.progress
import os

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    # TaskProgressColumn(),
    "{task.completed}/{task.total}",
    TimeRemainingColumn(),
)



with open('../SysMatConfig/Parameters.json') as json_file:
    parameters = json.load(json_file)

NImgX_ = parameters["numImageX"]
NImgY_ = parameters["numImageY"]
NDetY_ = parameters["pixelSiPM"]
NModule_ = parameters["numPanel"]
NDetX_ = parameters["numDetectorLayer"]

pixel_size_mm = 1
epsilon = 1e-13

cylinder_diameter_mm = NImgX_
cylinder_diameter_pixels = int(cylinder_diameter_mm / pixel_size_mm)


def reconstruct_image_MAP(phantomFile,filename):
    # Open the file.
    sysmatPath = parameters["sysmatPath"]
    dataSize = NImgX_*NImgY_*NDetY_*NModule_*NDetX_
    dataMatrix = []
    n_Rotations = parameters['numRotation']
    for idx_rotation in range(n_Rotations):
        inFname = 'sysmatMatrix_Rot_{0}_of_{1}.sysmat'.format(idx_rotation,n_Rotations)
        filePath = sysmatPath+inFname

        with rich.progress.open(filePath, 'rb') as inF:
            # Read in the matrix
            dataUnpack = np.asarray(struct.unpack('f'*dataSize, inF.read(dataSize*4)))
            # Reshape the 5D array into a 2D matrix
            dataMatrix.append(dataUnpack.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_)))
            # inF.close()

    dataMatrix = np.array(dataMatrix)
    dataMatrix = dataMatrix.reshape(-1, NImgX_*NImgY_)
    print("Complete Read-in Data!")
    print("{:>28}:\t{:}".format("Read-in System Matrix Shape", dataMatrix.shape))
    sysMatrix = dataMatrix[~np.all(dataMatrix == 0, axis=1)]

    dataUnpack = np.load(phantomFile)
    dataSize = NImgX_*NImgY_
    phantom = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))
    projection = np.matmul(sysMatrix, phantom.flatten())
    print("{:>28}:\t{:}".format("Projection Shape", projection.shape))

    NIteration = parameters["ReconstructionIterations"]
    reconImg = np.ones(NImgX_*NImgY_)
    sampleRate_ = 0.1
    totalSample = int(NIteration*sampleRate_)
    myFrames = np.zeros((totalSample, NImgX_, NImgY_))
    beta = 0.01 # weight or smoothening factor


    def backwardProj(reconImg, projection, sysMatrix):
        forwardLast = np.matmul(sysMatrix, reconImg)
        quotients = (projection[:] + epsilon)/(forwardLast[:] + epsilon)
        numerator = np.matmul(quotients, sysMatrix)
        denominator = np.sum(sysMatrix, axis=0)[:] + epsilon


        n, m = NImgX_, NImgY_
        reconImg = reconImg.reshape((NImgX_, NImgY_))
        potential_function_derivative = np.zeros_like(reconImg, dtype=np.float32)
        
        for i in range(1, n-1):
            for j in range(1, m-1):
                
                # finding the median of a 3x3 submatrix centered at (i,j)
                _3x3_mask = reconImg[i-1:i+2, j-1:j+2]
                median = np.median(_3x3_mask)
                
                # calculate the potential function's derivative using median method
                if median != 0:
                    potential_function_derivative[i][j] = beta * (reconImg[i][j] - median)/median


        reconImg = reconImg.flatten()
        denominator = denominator + potential_function_derivative.flatten()
        
        return numerator/denominator * reconImg



    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        # TaskProgressColumn(),
        "{task.completed}/{task.total}",
        TimeRemainingColumn(),
    )


    # Back Propogation iterations
    img_frames = []
    with progress:
        progress.console.print("Iterative reconstruction calculation...")
        task1 = progress.add_task("Iteration:", total=NIteration)
        
        for iter in range(NIteration):
            reconImg = backwardProj(reconImg, projection, sysMatrix)

            # storing it in a 2d matrix form
            img_frames.append(reconImg.reshape((NImgX_, NImgY_)))

            progress.advance(task1)
            if iter % 500 == 0:
                print(iter, "Iteration done")


    outFname = 'LIRImages/ReconPhantom/'+filename
    np.savez(outFname, myFrames[-1].astype(np.float32))

def reconstruct_image_MLEM(phantomFile,filename):

    sysmatPath = parameters["sysmatPath"]
    dataSize = NImgX_*NImgY_*NDetY_*NModule_*NDetX_
    dataMatrix = []
    n_Rotations = parameters['numRotation']
    for idx_rotation in range(n_Rotations):
        inFname = 'sysmatMatrix_Rot_{0}_of_{1}.sysmat'.format(idx_rotation,n_Rotations)
        filePath = os.path.join(sysmatPath, inFname)

        with rich.progress.open(filePath, 'rb') as inF:
            # Read in the matrix
            dataUnpack = np.asarray(struct.unpack('f'*dataSize, inF.read(dataSize*4)))
            # Reshape the 5D array into a 2D matrix
            dataMatrix.append(dataUnpack.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_)))
            # inF.close()


    dataMatrix = np.array(dataMatrix)
    dataMatrix = dataMatrix.reshape(-1, NImgX_*NImgY_)
    print("Complete Read-in Data!")


    imgTemplate = np.zeros((NImgX_, NImgY_))
    print("{:>28}:\t{:}".format("Read-in System Matrix Shape", dataMatrix.shape))

    # Remove zero rows from the matrix
    sysMatrix = dataMatrix[~np.all(dataMatrix == 0, axis=1)]
    print("{:>28}:\t{:}".format("Reduced System Matrix Shape", sysMatrix.shape))



    # Read in the phantom
    inFname = phantomFile
    dataUnpack = np.load(inFname)
    dataSize = NImgX_*NImgY_
    phantom = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))

    # Calculate forward projection
    projection = np.matmul(sysMatrix, phantom.flatten())
    print("{:>28}:\t{:}".format("Projection Shape", projection.shape))


    NIteration = parameters["ReconstructionIterations"]
    # reconImg = backwardProj(np.ones(NImgX_*NImgY_), projection, sysMatrix)
    reconImg = np.ones(NImgX_*NImgY_)
    

    def backwardProj(lastArr, projArr, sysMat):
        #reconImage, Projection, Sysmat
        forwardLast = np.matmul(sysMat, lastArr,dtype=np.float32)
        quotients = (projArr[:] + epsilon)/(forwardLast[:] + epsilon)
        return np.matmul(quotients, sysMat,dtype=np.float32)/(np.sum(sysMat, axis=0)[:] + epsilon)*lastArr


    def update(frame, myFrames, sampleRate, imshow_obj, pltTitle, cbar):
        # print("Generating Frame# {:5d}".format(frame), end='\r', flush=True)
        # print("Calculating Iteration# {:<5d}".format(frame), end='\r', flush=True)
        pltTitle.set_text(
            'Reconstructed Image Iteration#: {:>5d}'.format(int(frame/sampleRate)))
        imshow_obj.set_data(myFrames[frame])
        imshow_obj.norm.autoscale(imshow_obj._A)
        # imshow_obj.set_clim(0, 1)
        cbar.update_normal(imshow_obj.colorbar.mappable)

        return (imshow_obj, pltTitle,)


    sampleRate_ = 0.1
    totalSample = int(NIteration*sampleRate_)
    myFrames = np.zeros((totalSample, NImgX_, NImgY_))


    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        # TaskProgressColumn(),
        "{task.completed}/{task.total}",
        TimeRemainingColumn(),
    )


    with progress:
        progress.console.print("Iterative reconstruction calculation...")
        task1 = progress.add_task("Iteration:", total=NIteration)
        for iter in range(NIteration):
            reconImg = backwardProj(reconImg, projection, sysMatrix)
            if (iter*sampleRate_).is_integer():
                myFrames[int(iter*sampleRate_)] = reconImg.reshape((NImgX_, NImgY_))
                last_idx = int(iter*sampleRate_)
            progress.advance(task1)


    outFname = 'LIRImages/ReconPhantom/'+filename
    np.savez(outFname, myFrames[-1].astype(np.float32))




def calculate_LIR(delta):
    x_bg_file = 'LIRImages/ReconPhantom/phantom-background.npz'
    dataUnpack = np.load(x_bg_file)
    dataSize = NImgX_*NImgY_
    x_bg = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))

    x_bg_file = 'LIRImages/ReconPhantom/phantom-pointsource.npz'
    dataUnpack = np.load(x_bg_file)
    dataSize = NImgX_*NImgY_
    x_bg_ps = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))

    matrix = (x_bg_ps - x_bg) / delta

    return matrix

def calculate_CRC(LIR, point_source_x, point_source_y):
    return LIR[point_source_x,point_source_y]



phantom = np.zeros((NImgX_, NImgY_))


cylinder_center_x = NImgX_ // 2
cylinder_center_y = NImgY_ // 2
cylinder_radius = cylinder_diameter_pixels // 2
for i in range(NImgX_):
    for j in range(NImgY_):
        if (i - cylinder_center_x) ** 2 + (j - cylinder_center_y) ** 2 <= cylinder_radius ** 2:
            phantom[i, j] = 1

# for i in range(NImgX_):
#     for j in range(NImgY_):
#         if (i - cylinder_center_x) ** 2 + (j - cylinder_center_y) ** 2 <= 5 ** 2:
#             phantom[i, j] = 5


outFname = 'LIRImages/Phantom/phantom-background.npz'
fname = 'phantom-background.npz'
np.savez(outFname,phantom.astype(np.float32))
recon_file = 'LIRImages/ReconPhantom/'+fname
npz_file1 = np.load(outFname)

npz_data = np.load(outFname)
# Save individual arrays as a .mat file
data = {}
for name, arr in npz_data.items():
    data[name] = arr
import scipy.io as sio
sio.savemat('loaded_data.mat', data)

reconstruct_image_MAP(outFname,fname)
dataUnpack = np.load(recon_file)
dataSize = NImgX_*NImgY_
x_bg = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))
# for i in range(NImgX_):
#     for j in range(NImgY_):
#         print(x_bg[i][j])
npz_file3 = np.load(recon_file)

delta = 5
point_source_location_mm = 5
point_source_location_pixels = int(point_source_location_mm / pixel_size_mm)
point_source_x = NImgX_ // 2 + point_source_location_pixels
point_source_y = NImgY_ // 2
phantom[point_source_x, point_source_y] = delta

outFname = 'LIRImages/Phantom/phantom-pointsource.npz'
fname = 'phantom-pointsource.npz'
npz_file2 = np.load(outFname)
np.savez(outFname,phantom.astype(np.float32))
recon_file = 'LIRImages/ReconPhantom/'+fname
reconstruct_image_MAP(outFname,fname)
dataUnpack = np.load(recon_file)
dataSize = NImgX_*NImgY_
x_bg_ps = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))
# for i in range(NImgX_):
#     for j in range(NImgY_):
#         print(x_bg_ps[i][j])
npz_file4 = np.load(recon_file)


LIR = calculate_LIR(delta)

CRC = calculate_CRC(LIR, point_source_x, point_source_y)
print('CRC = ',CRC)




image1 = npz_file1['arr_0']
image2 = npz_file2['arr_0']
image3 = npz_file3['arr_0']
image4 = npz_file4['arr_0']
image1 = image1.reshape(180, 180)
image2 = image2.reshape(180, 180)
image3 = image3.reshape(180, 180)
image4 = image4.reshape(180, 180)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(image1, cmap='turbo')
axes[0, 0].set_title('Background Phantom')
axes[0, 0].axis('off')

axes[0, 1].imshow(image2, cmap='turbo')
axes[0, 1].set_title('Phantom with Point Source')
axes[0, 1].axis('off')

axes[1, 0].imshow(image3, cmap='turbo')
axes[1, 0].set_title('Reconstructed BP')
axes[1, 0].axis('off')

axes[1, 1].imshow(image4, cmap='turbo')
axes[1, 1].set_title('Reconstructed BP w Point Source')
axes[1, 1].axis('off')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
combined_image = np.vstack([np.hstack([image1, image2]), np.hstack([image3, image4])])

plt.savefig('LIRImages/combined_image.png',bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.imshow(combined_image, cmap='turbo')
plt.axis('off')
plt.show()
