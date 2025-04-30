from tkinter import E
import numpy as np
from matplotlib import pyplot as plt
import time
from skimage.filters import gaussian
from skimage.transform import resize


def imsaveWrapper(img, name):
    temp = np.copy(img) / max(img.max(), 1)
    if len(img.shape) == 2:
        plt.imsave(f'{name}.png', np.stack((temp, temp, temp), axis=2))
    else:
        plt.imsave(f'{name}.png', temp)

def resizeWrapper(path, factor):
    img = plt.imread(path + '.png')
    HEIGHT, WIDTH, _ = img.shape

    R = resize(img[:, :, 0], (np.ceil(HEIGHT * factor), np.ceil(WIDTH * factor)), anti_aliasing=False)
    R = resize(R, (HEIGHT, WIDTH), anti_aliasing=False)
    G = resize(img[:, :, 1], (np.ceil(HEIGHT * factor), np.ceil(WIDTH * factor)), anti_aliasing=False)
    G = resize(G, (HEIGHT, WIDTH), anti_aliasing=False)
    B = resize(img[:, :, 2], (np.ceil(HEIGHT * factor), np.ceil(WIDTH * factor)), anti_aliasing=False)
    B = resize(B, (HEIGHT, WIDTH), anti_aliasing=False)

    return np.stack((R, G, B), axis=2)

def gaussianWrapper(path, sigma):
    img = rgb2gray(plt.imread(path + '.png'))
    img = gaussian(img, sigma)
    plt.imsave(f'{path}.png', np.stack((img, img, img), axis=2))

def rgb2gray(img):
    rgb = np.sum(img, axis=2) / 3
    return rgb / rgb.max()

def SSD(x, y, axes):
    return np.sum((x - y) ** 2, axis=axes)

def NCC(x, y, axes):
    mean_x = np.mean(x, axis=axes)
    mean_y = np.mean(y, axis=(axes[0] - 1, axes[1] - 1))

    sigma_x = np.std(x, axis=axes)
    sigma_y = np.std(y, axis=(axes[0] - 1, axes[1] - 1))

    return np.sum((x - mean_x[:, :, np.newaxis, np.newaxis]) * (y - mean_y[:, np.newaxis, np.newaxis]), axis = axes) / (sigma_x * sigma_y) / (x.shape[-2] * x.shape[-1])

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

        # Format dynamic programming matrix
        dr = (0, min(63, WIDTH - 1)) # inclusive

        # Fill in matrix column by column
        occlusionConstant = 0.5

        # First row
        for col in range(1, dr[1] - dr[0] + 1):
            DSI[0, col] = occlusionConstant * col

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

        curr[leftOccluded[:, col]] = np.maximum(prev[leftOccluded[:, col]], curr[leftOccluded[:, col]])

    for col in range(WIDTH - 2, -1, -1):
        prev = depth[:, col + 1]
        curr = depth[:, col]
        curr[rightOccluded[:, col]] = np.maximum(prev[rightOccluded[:, col]], curr[rightOccluded[:, col]])
        
    imsaveWrapper(depth, f'../images/{name}/depth')


names = ['art', 'books', 'computer', 'dolls', 'drumsticks', 'dwarves', 'laundry', 'moebius', 'reindeer']

for name in names[-1:]:
    w = 15
    img1 = rgb2gray(plt.imread(f"../images/{name}/left.png"))
    img2 = rgb2gray(plt.imread(f"../images/{name}/right.png"))
    assert(img1.shape == img2.shape)
    dynamic(img1, img2, w)

'''

for name in names:
    img2 = gaussianWrapper(f"../images/{name}/depth", 0.6);

'''


