import modal
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import io

# 1. Define the Modal App and the Environment Image
app = modal.App("skytree-bg-remove")
image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers==4.39.3", "Pillow", "fastapi[standard]", "scikit-image"
)

# 2. Define the FastAPI App (This handles the web requests)
web_app = FastAPI()

# Allow the React app to communicate with this API
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Create a Modal Class to hold the heavy AI Model 
# (This keeps the model loaded in GPU memory so it doesn't have to reload on every request!)
@app.cls(gpu="T4", image=image)
class Model:
    @modal.enter()
    def load_model(self):
        print("Loading RMBG-1.4 model into GPU memory...")
        from transformers import pipeline
        self.pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=0)
        print("Model loaded successfully!")

    @modal.method()
    def predict(self, image_bytes: bytes) -> bytes:
        from PIL import Image
        print("Processing new image...")
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output_image = self.pipe(input_image)
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        print("Image processed successfully!")
        return img_byte_arr.getvalue()

# 4. Connect FastAPI to the Modal Class
@web_app.post("/remove_bg")
async def remove_bg(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Call the GPU function remotely!
    model = Model()
    result_bytes = await model.predict.remote.aio(contents)
    
    return Response(content=result_bytes, media_type="image/png")

# 5. Expose the FastAPI app to the public web
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
