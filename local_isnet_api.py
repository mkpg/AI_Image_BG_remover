from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import io
from rembg import remove, new_session
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize session as None so it doesn't block startup
session = None

@app.post("/remove_bg")
async def remove_bg(file: UploadFile = File(...)):
    global session
    if session is None:
        print("First request: Downloading/Loading model into cache...")
        session = new_session("isnet-general-use")
        
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))
    
    # Process the image (Alpha matting disabled due to ghosting artifacts on complex backgrounds)
    output_image = remove(input_image, session=session)
    
    # Return as PNG
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
