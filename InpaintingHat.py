import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os

# ==========================================
# --- Environment Setup & SDXL Model Initialization ---
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
use_vram_optimization = True  
model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

print(f"[{device.upper()}] Initializing device and SDXL pipeline...")

try:
    pipeline = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None
    )
    pipeline = pipeline.to(device)

    if use_vram_optimization and device == "cuda":
        pipeline.enable_model_cpu_offload()
        print("VRAM optimization (model CPU offload) enabled.")

except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# ==========================================
# --- Core SDXL Inpainting Pipeline ---
# ==========================================
def run_inpainting_pipeline(
    pipe, 
    prompt: str, 
    negative_prompt: str, 
    init_image: Image.Image,
    mask_image: Image.Image, 
    max_size: int = 1024,      # Optimal resolution for SDXL computation
    up_shift: int = 150,       # [Core] Pixels to shift the mask upwards (secures structural volume)
    dilation_kernel: int = 50, # Kernel size for overall mask dilation
    blur_radius: int = 80      # Blur radius for boundary blending during composition
):
    print("------------------------------------------")
    print("Starting high-resolution SDXL inpainting (Volume Expansion Mode)...")
    
    orig_w, orig_h = init_image.size

    # ------------------------------------------
    # 1. Mask Transformation & Expansion
    # ------------------------------------------
    mask_np = np.array(mask_image.convert("L"))
    h, w = mask_np.shape
    
    # Transformation matrix to shift the mask upwards (negative y-axis translation)
    m_matrix = np.float32([[1, 0, 0], [0, 1, -up_shift]])
    shifted_mask = cv2.warpAffine(mask_np, m_matrix, (w, h))
    
    # Merge original mask and shifted mask (creates a vertically elongated mask)
    combined_mask = cv2.bitwise_or(mask_np, shifted_mask)
    
    # Apply morphological dilation to secure boundary context for the model
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    final_mask_np = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Convert back to PIL Image object
    expanded_mask_image = Image.fromarray(final_mask_np)

    # ------------------------------------------
    # 2. Safe Resizing for SDXL Compatibility (Multiples of 8)
    # ------------------------------------------
    ratio = max_size / max(orig_w, orig_h)
    new_w = int((orig_w * ratio) // 8 * 8)
    new_h = int((orig_h * ratio) // 8 * 8)
    
    resized_img = init_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_mask = expanded_mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)
    resized_mask = resized_mask.point(lambda p: 255 if p > 128 else 0)
    
    print(f"Mask expanded upwards by {up_shift}px. Compute resolution: ({new_w}x{new_h})")

    # ------------------------------------------
    # 3. SDXL Model Inference
    # ------------------------------------------
    print("Running SDXL inference... (Generating structural volume)")
    with torch.no_grad():
        inpainted_img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_img,
            mask_image=resized_mask,
            guidance_scale=8.5, 
            num_inference_steps=50,
        ).images[0]

    # ------------------------------------------
    # 4. High-Res Restoration & Smooth Compositing
    # ------------------------------------------
    print("Restoring original resolution and compositing with the background...")
    restored_img = inpainted_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    # Apply heavy Gaussian blur to the expanded mask to create a soft alpha channel
    strict_mask = expanded_mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Smoothly composite the newly generated region over the original image
    final_result = Image.composite(restored_img, init_image, strict_mask)
    
    print("Inpainting process completed successfully.")
    print("------------------------------------------")
    return final_result


# ==========================================
# --- Execution / Main ---
# ==========================================
if __name__ == "__main__":
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    image_path = os.path.join(test_dir, "origin.jpg")
    mask_path = os.path.join(test_dir, "mask.png")
    output_path = os.path.join(test_dir, "result_sdxl.png")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Error: Please place 'origin.jpg' and 'mask.png' inside the '{test_dir}' directory.")
        exit()

    origin_image = Image.open(image_path).convert("RGB")
    origin_mask = Image.open(mask_path).convert("L")

    # Volume-enhancing prompt engineering
    positive_prompt = "voluminous hair, thick hair, fluffy hairstyle, highly detailed brown hair texture, extremely realistic hair strands, natural lighting, blending perfectly, sharp focus, 8k, masterpiece"
    negative_prompt = "flat hair, bald, hat, cap, plastic texture, cartoon, illustration, bad hair blending, unnatural pattern, artifacts, weird shape, blurring"

    # Execute pipeline
    final_image = run_inpainting_pipeline(
        pipe=pipeline,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        init_image=origin_image,
        mask_image=origin_mask,
        max_size=2048,      
        up_shift=80,       # Increase to 100~150 if the generated hair look flat.
        dilation_kernel=30, # Increase to 80~100 if artifacts or original edges (e.g., hat) remain visible.
        blur_radius=8      # Increase to 10~20 for smoother boundary blending.
    )

    final_image.save(output_path)
    print(f"SDXL output successfully saved to '{output_path}'.")