from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    )
import torch
from PIL import Image
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

INPUT_IMAGE_PATH = args.folder
MASK_IMAGE_PATH = "tmp/masks"
SAVE_IMAGE_PATH = "tmp/steo"

os.makedirs(SAVE_IMAGE_PATH,exist_ok   = True)

model_path = "onkarsus13/controlnet_stablediffusion_scenetextEraser"

device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained(model_path)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

generator = torch.Generator(device).manual_seed(1)
for image_name in tqdm(os.listdir(MASK_IMAGE_PATH)):
    image = Image.open(os.path.join(INPUT_IMAGE_PATH,image_name))
    mask_image = Image.open(os.path.join(MASK_IMAGE_PATH,image_name))

    original_image_size = image.size
    new_image_size = (512, 512)

    image = image.resize(new_image_size)
    mask_image = mask_image.resize(new_image_size)

    result_image = pipe(
        image,
        mask_image,
        [mask_image],
        num_inference_steps=40,
        generator=generator,
        controlnet_conditioning_scale=1.0,
        guidance_scale=1.0
    ).images[0]

    result_image = result_image.resize(original_image_size)
    result_image.save(os.path.join(SAVE_IMAGE_PATH,image_name))

print(f"<<<<<파일 확인>>>>> scene_text_eraser.py")

