from fastapi import FastAPI
from fastapi.responses import Response

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import io

app = FastAPI()


@app.get("/")
async def root():
    # Disparity would contain the output of the stereo matching algorithm
    # Must be grayscale with dtype of uint8
    disparity = (plt.imread("image.png") * 255).astype(np.uint8)[:, :, 0]

    imgByteArr = io.BytesIO()
    Image.fromarray(disparity, "L").save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()

    return Response(content=imgByteArr, media_type="image/png")
