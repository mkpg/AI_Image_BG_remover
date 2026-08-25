import gradio as gr
from transformers import pipeline
import spaces

# --- BUG FIX FOR NEW TRANSFORMERS VERSIONS ---
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
# ---------------------------------------------

print("Loading Enterprise AI Model (briaai/RMBG-1.4)...")
# We no longer force CPU! We will let it use the massive ZeroGPU!
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# ZeroGPU strictly requires this decorator to allow GPU access
@spaces.GPU
def remove_bg(image):
    print("Processing image on ZeroGPU...")
    output_image = pipe(image)
    return output_image

demo = gr.Interface(
    fn=remove_bg, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Image(type="pil", format="png"), 
    title="🟢 Skytree AI API is Online!",
    description="Your API is running perfectly on the ZeroGPU hardware."
)

if __name__ == "__main__":
    demo.launch()
