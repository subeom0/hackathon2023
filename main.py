import torch
from torch import nn
import numpy as np
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "frontend-app.yourdomain.com",
    "frontend-app2.yourdomain.com:7000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    user: list
    items: list

# json
# {
#     "user":[1,0,0,1,0,0,1,1,1,1,1,1], -> 12개
#     "items":
#         {
#             "id":"1qdoqfbhsdjbvsb",
#             "env":[[0,1,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,0]] -> 28개
#         }
    
# }
 

class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


@app.post("/return-accuracy/")
def model_out(item: Item):
    datas = []
    for json in item.items:
        datas.append([item.user + json["env"], json["id"]])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    

    def predict(data, model = torch.load('model.pth')):
        data = data.to(device)
        model.eval()
        return model(data).tolist()[0][0]

    accuracy = [] 
    for data in datas:
        accuracy.append((predict(torch.from_numpy(np.array([list(map(int,data[0]))])).type(torch.FloatTensor)),data[1]))
    return sorted(accuracy, reverse=True)



if __name__ == '__main__':
    uvicorn.run(app)