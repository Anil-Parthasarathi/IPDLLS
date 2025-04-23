from tkinter import E
import numpy as np
from matplotlib import pyplot as plt
import time
from skimage.filters import gaussian, sobel


def imsaveWrapper(img, name):
    temp = img / max(img.max(), 1)
    if len(img.shape) == 2:
        plt.imsave(f'./results/{name}.png', np.stack((temp, temp, temp), axis=2))
    else:
        plt.imsave(f'./results/{name}.png', temp)

def rgb2gray(img):
    rgb = np.sum(img, axis=2) / 3
    return rgb / rgb.max()

def SSD(x, y, axes):
    return np.sum((x - y) ** 2, axis=axes)

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
    for outerRow in range(HEIGHT):
        DSI = np.zeros((WIDTH, WIDTH)).astype(img1.dtype)
        DSI += SSD(window1[outerRow][:, np.newaxis], window2[outerRow], (2, 3)).T
        DSI /= DSI.max()
        imsaveWrapper(DSI, 'dsi_pre')

        # Format dynamic programming matrix
        dr = (0, min(64, WIDTH - 1)) # inclusive
        # diagIndices = np.add.outer(-np.arange(WIDTH), np.arange(WIDTH)) 
        # diagIndices = np.add.outer(-np.arange(WIDTH + (dr[1] - dr[0])), np.arange(WIDTH + (dr[1] - dr[0]))) 
        # diagIndices = ((diagIndices >= dr[0]) & (diagIndices <= dr[1]))[:DSI.shape[0]]
        # DSI = np.concatenate((DSI, np.zeros((DSI.shape[0], dr[1] - dr[0]))), axis=1)[diagIndices]
        # DSI = DSI.reshape(WIDTH, dr[1] - dr[0] + 1).T
        # imsaveWrapper(DSI, 'dsi_post')

        # Fill in matrix column by column
        occlusionConstant = 0.001
        # print(occlusionConstant, DSI)

        # First row
        for col in range(1, dr[1] - dr[0] + 1):
            DSI[0, col] = min(DSI[0, col], DSI[0, col - 1] + occlusionConstant)

        # Remaining rows
        baseCol = 1
        counter = 1
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

        imsaveWrapper(DSI, 'mid')

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
                col -= 1
            elif (currMin == occlusionRight):
                row -= 1

            assert(col >= row)

        depth[outerRow, 0] = 0

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

rows = 500
cols = 500
w = 11
img1 = rgb2gray(plt.imread("data/view1.png"))[: rows, : cols]
img2 = rgb2gray(plt.imread("data/view5.png"))[: rows, : cols]
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
