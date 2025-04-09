from fastapi import FastAPI, Response
from matplotlib import pyplot as plt

app = FastAPI()






@app.get("/")
async def root():
    path = "./data/8192.png"
    image = plt.imread(path).tobytes
    print(image)
    return {"Message": "hei"}
    # return Response(content=image, media_type="binary/octet-stream")