import modal
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
import io

app = modal.App("skytree-bg-remove-isnet")

# We just need rembg, Pillow, and fastapi. 
# rembg automatically downloads the isnet-general-use model on first run!
image = modal.Image.debian_slim().pip_install(
    "rembg", "Pillow", "fastapi[standard]"
)

from fastapi import FastAPI, Response, Request
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/")
async def remove_bg(request: Request):
    from rembg import remove, new_session
    from PIL import Image
    
    contents = await request.body()
    input_image = Image.open(io.BytesIO(contents))
    session = new_session("isnet-general-use")
    output_image = remove(input_image, session=session)
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
