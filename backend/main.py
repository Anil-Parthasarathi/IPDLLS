from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import io

app = FastAPI()

origins = [
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


@app.get("/disparity")
async def disparity(name: str):
    try:
        left = plt.imread(f"../images/{name}/left.png")
        right = plt.imread(f"../images/{name}/right.png")
    except FileNotFoundError:
        raise HTTPException(status_code=422, detail="Image does not exist")

    # Disparity would contain the output of the stereo matching algorithm
    # Must be grayscale with dtype of uint8
    disparity = (np.concatenate((left, right)) * 255).astype(np.uint8)[:, :, 0]  # UPDATE

    # Convert 2D to bytes of png
    imgByteArr = io.BytesIO()
    Image.fromarray(disparity, "L").save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()

    return Response(content=imgByteArr, media_type="image/png")
