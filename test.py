import torch 

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else: device = "cpu"
print(f"using device: {device}")

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model,confidence_threshold=0.7)

image_path = "image/example_image8.jpg"
image = Image.open(image_path).convert("RGB")
inference_state = processor.set_image(image)

text_prompt = "a car"
output = processor.set_text_prompt(
    state=inference_state,
    prompt=text_prompt
)

img0 = Image.open(image_path)
plot_results(img0, inference_state)
plt.savefig("result_output.jpg")

# masks = output["masks"]
# boxes = output["boxes"]

# import numpy as np

# plt.figure(figsize=(10,10))
# plt.imshow(image)

# if len(masks) > 0:
#     for i in range(len(masks)):
#         mask = masks[i].cpu().numpy()
#         if mask.ndim == 3: mask = mask[0]
    
#         color = np.concatenate([np.random.random(3), [0.5]])

#         mask_image = np.zeros((*mask.shape, 4))
#         mask_image[mask > 0] = color
#         plt.gca().imshow(mask_image)

#         box = boxes[i].cpu().numpy()
#         plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
#                                         edgecolor='green', facecolor='none', lw=2))
    
#     plt.title(f"Prompt: {text_prompt}")
#     plt.axis('off')
#     plt.savefig("result.png")
#     plt.show()
#     print("Done")
# else:
#     print("None")
