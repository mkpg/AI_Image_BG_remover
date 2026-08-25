import modal
app = modal.App("test-bg")
image = modal.Image.debian_slim().pip_install("torch", "torchvision", "transformers==4.39.3", "Pillow", "scikit-image")
@app.function(image=image)
def test():
    from transformers import pipeline
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu")
    print("Success 4.39.3!")
