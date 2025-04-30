import sys
import os.path
import numpy as np
from matplotlib import pyplot as plt


def imsaveWrapper(img, name):
    temp = np.copy(img) / max(img.max(), 1)
    if len(img.shape) == 2:
        plt.imsave(f'{name}.png', np.stack((temp, temp, temp), axis=2))
    else:
        plt.imsave(f'{name}.png', temp)

def rgb2gray(img):
    rgb = np.sum(img, axis=2) / 3
    return rgb / rgb.max()

def NCC(x, y, axes):
    mean_x = np.mean(x, axis=axes)
    mean_y = np.mean(y, axis=(axes[0] - 1, axes[1] - 1))

    sigma_x = np.std(x, axis=axes)
    sigma_y = np.std(y, axis=(axes[0] - 1, axes[1] - 1))

    return np.sum((x - mean_x[:, :, np.newaxis, np.newaxis]) * (y - mean_y[:, np.newaxis, np.newaxis]), axis = axes) / (sigma_x * sigma_y) / (x.shape[-2] * x.shape[-1])

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
        
    return depth


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Must enter path to directory containing left.png and right.png and window size")
        exit(1)

    img1 = rgb2gray(plt.imread(os.path.join(sys.argv[1], "left.png")))
    img2 = rgb2gray(plt.imread(os.path.join(sys.argv[1], "right.png")))

    imsaveWrapper(dynamic(img1, img2, int(sys.argv[2])), os.path.join(sys.argv[1], "depth"))
