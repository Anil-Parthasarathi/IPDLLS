from tkinter import E
import numpy as np
from matplotlib import pyplot as plt
import time
from skimage.filters import gaussian, sobel


def imsaveWrapper(img, name):
    temp = np.copy(img) / max(img.max(), 1)
    if len(img.shape) == 2:
        plt.imsave(f'./results/{name}.png', np.stack((temp, temp, temp), axis=2))
    else:
        plt.imsave(f'./results/{name}.png', temp)

def rgb2gray(img):
    rgb = np.sum(img, axis=2) / 3
    return rgb / rgb.max()

def SSD(x, y, axes):
    return np.sum((x - y) ** 2, axis=axes)

def NCC(x, y, axes):
    # for i in range(5):
    #     tempx = np.copy(x[i, 0])
    #     tempy = np.copy(y[i])
    #     # print(tempx)
    #     mean_x = np.mean(tempx)#), axis=axes)
    #     mean_y = np.mean(tempy)
    #     # print(mean_x)
    #     # print(mean_x)

    #     sigma_x = np.std(tempx)#, axis=axes)
    #     sigma_y = np.std(tempy)
    #     # print(sigma_x)

    #     val = np.sum((tempx - mean_x) * (tempy - mean_y)) / (sigma_x * sigma_y * tempx.size) 
    #     # print(val)
        # if (abs(val) > 1):
        #     break


    mean_x = np.mean(x, axis=axes)
    mean_y = np.mean(y, axis=(axes[0] - 1, axes[1] - 1))
    # print(mean_x[:5])
    # print(mean_x.shape, mean_y.shape)

    sigma_x = np.std(x, axis=axes)
    sigma_y = np.std(y, axis=(axes[0] - 1, axes[1] - 1))
    # print(sigma_x[:5])
    # print(sigma_x.shape, sigma_y.shape)
    # exit(-1)

    return np.sum((x - mean_x[:, :, np.newaxis, np.newaxis]) * (y - mean_y[:, np.newaxis, np.newaxis]), axis = axes) / (sigma_x * sigma_y) / (x.shape[-2] * x.shape[-1])
    # print(temp.shape)
    # exit(0)

# Inputs must be grayscale (and same size)
# Window must be odd dimensions
def two(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.zeros_like(img1)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")

    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            depth[row, col] = np.argmin(SSD(window1[row, col], window2[row], (1, 2))) - col

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_two.jpg', depth / depth.max())

def one(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.ones_like(img1)
    depth[:, 0] = 0
    depth = np.cumsum(depth, axis=1, dtype=img1.dtype)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")
    
    # Calculate Disparities
    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))
    for row in range(HEIGHT):
        depth[row] -= np.argmin(SSD(window1[row][:, np.newaxis], window2[row], (2, 3)), axis=1)

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_one.jpg', depth / depth.max())

def zero(img1, img2, windowDim): 
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.ones_like(img1)
    depth[:, 0] = 0
    depth = np.cumsum(depth, axis=1, dtype=img1.dtype)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")

    # Calculate Disparities
    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))

    depth -= np.argmin(SSD(window1[:, :, np.newaxis], window2[:, np.newaxis], (3, 4)), axis=2)

    depth = abs(depth)
    depth = np.stack((depth, depth, depth), axis=-1)
    plt.imsave('./results/0_zero.jpg', depth / depth.max())

def dynamic(img1, img2, windowDim):
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]

    depth = np.zeros_like(img1)

    # Pad both images to accomodate window
    pad = windowDim // 2
    pad = ((pad, pad), (pad, pad))
    img1 = np.pad(img1, pad, "symmetric")
    img2 = np.pad(img2, pad, "symmetric")
    
    window1 = np.lib.stride_tricks.sliding_window_view(img1, (windowDim, windowDim))
    window2 = np.lib.stride_tricks.sliding_window_view(img2, (windowDim, windowDim))

    # col is window row is comparison
    depth = np.zeros((HEIGHT, WIDTH))
    leftOccluded = np.zeros((HEIGHT, WIDTH)).astype(np.bool_)
    rightOccluded = np.zeros((HEIGHT, WIDTH)).astype(np.bool_)
    for outerRow in range(HEIGHT // 1):
        DSI = np.ones((WIDTH, WIDTH)).astype(img1.dtype)
        DSI -= NCC(window1[outerRow][:, np.newaxis], window2[outerRow], (2, 3)).T
        # print(DSI.max(), DSI.min())

        # Format dynamic programming matrix
        dr = (0, min(63, WIDTH - 1)) # inclusive
        # diagIndices = np.add.outer(-np.arange(WIDTH), np.arange(WIDTH)) 
        # diagIndices = np.add.outer(-np.arange(WIDTH + (dr[1] - dr[0])), np.arange(WIDTH + (dr[1] - dr[0]))) 
        # diagIndices = ((diagIndices >= dr[0]) & (diagIndices <= dr[1]))[:DSI.shape[0]]
        # DSI = np.concatenate((DSI, np.zeros((DSI.shape[0], dr[1] - dr[0]))), axis=1)[diagIndices]
        # DSI = DSI.reshape(WIDTH, dr[1] - dr[0] + 1).T
        # imsaveWrapper(DSI, 'dsi_post')

        # Fill in matrix column by column
        occlusionConstant = .25
        # print(occlusionConstant, DSI)

        # First row
        for col in range(1, dr[1] - dr[0] + 1):
            DSI[0, col] = occlusionConstant * col #min(DSI[0, col], DSI[0, col - 1] + occlusionConstant)

        # Remaining rows
        baseCol = 1
        for row in range(1, DSI.shape[0]):
            for col in range(dr[1] - dr[0] + 1):
                if (baseCol + col >= DSI.shape[1]):
                    break

                dissimilarity = DSI[row - 1, baseCol + col - 1] + DSI[row, baseCol + col]
                occlusionLeft = dissimilarity
                occlusionRight = dissimilarity
                    
                if (col > 0):
                    occlusionLeft = DSI[row, baseCol + col - 1] + occlusionConstant
                if (col + 1 < dr[1] - dr[0] + 1):
                    occlusionRight = DSI[row - 1, baseCol + col] + occlusionConstant

                DSI[row, baseCol + col] = min(dissimilarity, occlusionLeft, occlusionRight) 

            baseCol += 1

        # imsaveWrapper(DSI, 'mid')

        # Backtrace
        col = DSI.shape[1] - 1
        row = DSI.shape[0] - 1
        while (row > 0 and col > 0):
            dissimilarity = DSI[row - 1, col - 1]
            occlusionLeft = dissimilarity
            occlusionRight = dissimilarity

            # print(col - row)
            if (col - row > 0):
                occlusionLeft = DSI[row, col - 1]
            if (col - row < dr[1] - dr[0] + 1):
                occlusionRight = DSI[row - 1, col]
                    
            currMin = min(dissimilarity, occlusionLeft, occlusionRight)

            if (currMin == dissimilarity):
                depth[outerRow, col] = col - row
                col -= 1
                row -= 1
            elif (currMin == occlusionLeft):
                leftOccluded[outerRow, col] = True
                col -= 1
            elif (currMin == occlusionRight):
                rightOccluded[outerRow, col] = True
                row -= 1

            assert(col >= row)

    # Fill in Occlusions
    for col in range(1, WIDTH):
        prev = depth[:, col - 1]
        curr = depth[:, col]

        # if (np.any(leftOccluded[:, col])):
        curr[leftOccluded[:, col]] = np.maximum(prev[leftOccluded[:, col]], curr[leftOccluded[:, col]])

    for col in range(WIDTH - 2, -1, -1):
        prev = depth[:, col + 1]
        curr = depth[:, col]

        # if (np.any(rightOccluded[:, col])):
        curr[rightOccluded[:, col]] = np.maximum(prev[rightOccluded[:, col]], curr[rightOccluded[:, col]])

        # # Initial row and col
        # for i in range(DSI.shape[0]):
        #     DSI[i, 0] = i * occlusionConstant

        # for i in range(DSI.shape[1]):
        #     DSI[0, i] = i * occlusionConstant

        # # Remaining columns
        # for row in range(1, DSI.shape[0]):
        #     for col in range(1, DSI.shape[1]):

        #         dissimilarity = DSI[row - 1, col - 1] + DSI[row, col]
        #         occlusionLeft =  DSI[row - 1, col] + occlusionConstant
        #         occlusionRight = DSI[row, col - 1] + occlusionConstant

        #         DSI[row][col] = min(dissimilarity, occlusionLeft, occlusionRight) 

        # # Work backwards
        # row = DSI.shape[0] - 1
        # col = DSI.shape[1] - 1
        # while (row != 0 and col != 0):

        #     dissimilarity = DSI[row - 1, col - 1]
        #     occlusionLeft = DSI[row - 1, col]
        #     occlusionRight = DSI[row, col - 1]
                    
        #     currMin = min(dissimilarity, occlusionLeft, occlusionRight)

        #     if (currMin == dissimilarity): # Not necessary unless there is a tie (included to avoid unnecessary occlussion)
        #         depth[outerRow, col] = row
        #         row -= 1
        #         col -= 1
        #     elif (currMin == occlusionLeft):
        #         row -= 1
        #     elif (currMin == occlusionRight):
        #         col -= 1

        # depth[outerRow, 0] = row
        

    # print(DSI.max(), min(np.inf, 39999999))
    # imsaveWrapper(DSI, 'DSI')
    imsaveWrapper(depth, 'depth')
    # DSI = DSI.reshape(2, WIDTH//2)
    # DSI = DSI.diagonal(disparity_range[1])
    # print(DSI.shape)

# A = np.array([
#     [0,0,0,0],
#     [0,4,0,0],
#     [0,0,0,1]
# ])
# B = np.array([
#     [0,0,0,0],
#     [0,1,4,0],
#     [0,0,0,0]   
# ])
# base(A, B, 3)
# print('--------------------')
# working(A, B, 3)
# print('--------------------')
# zero(A, B, 3)
# print('--------------------')
# dynamic(A, B, 3)

rows = 375
cols = 450
w = 5
img1 = rgb2gray(plt.imread("data/img2.png"))[: rows, : cols]
img2 = rgb2gray(plt.imread("data/img6.png"))[: rows, : cols]
dynamic(img1, img2, w)


# print("Img shape:", img1.shape, "Window:", w)
# print("Two for loops time: ", end='', flush=True)
# start = time.time()
# two(img1, img2, w)
# print(time.time() - start)

# print("One for loops time: ", end='', flush=True)
# start = time.time()
# one(img1, img2, w)
# print(time.time() - start)

# print("Zero for loops time: ", end='', flush=True)
# start = time.time()
# zero(img1, img2, w)
# print(time.time() - start)
