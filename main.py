import io
import base64
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager


# --- 1. MODEL ARCHITECTURE ---
class MyNN(nn.Module):
    def __init__(self, input_features=1):
        super().__init__()

        self.feautures = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.feautures(x)
        x = self.classifier(x)
        return x


# Global model reference
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    import __main__
    setattr(__main__, "MyNN", MyNN)
    sys.modules['__main__'].MyNN = MyNN

    # Instantiate model
    model = MyNN(input_features=1).to(device)

    # Safely load weights/model
    checkpoint = torch.load("digit_model.pt", map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    
    yield


app = FastAPI(title="Digit Recognizer API", lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 3. PREPROCESSING PIPELINE ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class ImagePayload(BaseModel):
    image_data: str


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict_digit(payload: ImagePayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded yet.")

    try:
        base64_str = payload.image_data
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)
        pil_img = Image.open(io.BytesIO(image_bytes))

        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            "prediction": int(predicted.item()),
            "confidence": round(float(confidence.item()) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)