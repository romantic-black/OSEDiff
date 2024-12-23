from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import glob
from typing import List
from PIL import Image
import torch
from osediff import OSEDiff_test
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import argparse
from torchvision import transforms
import numpy as np

neg_prompt = ['man', 'crowded', 'person', 'walk','pick up','skate', 'skier', 'woman',
                                  'drive', 'car','taxi', 'jeep', 'vehicle', 'suv', 'minivan',
                                  'van', 'snowstorm', 'snow', 'snowy', 'blue', 'blanket', 'toy car', 'motorcycle', 'park', 'parking garage',
                                  'white','black',  'fill', 'night', 'rainy', 'rain']

# Define transformations
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize FastAPI app
app = FastAPI()

# Define default arguments
class Args:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    seed = 42
    process_size = 512
    upscale = 1
    align_method = 'adain'
    osediff_path = '/mnt/e/Output/osediff/model_32001.pkl'
    prompt = ''
    ram_path = "/mnt/e/download/ram_swin_large_14m.pth"
    ram_ft_path = None
    save_prompts = True
    mixed_precision = 'fp16'
    merge_and_unload_lora = False
    vae_decoder_tiled_size = 224
    vae_encoder_tiled_size = 1024
    latent_tiled_size = 96
    latent_tiled_overlap = 32

# Create an instance of Args
args = Args()

# Global variables for model
model = None
DAPE = None
weight_dtype = torch.float32


# Load model
@app.get("/load_model")
def load_model():
    global model, DAPE, weight_dtype

    # Load OSEDiff model
    model = OSEDiff_test(args)

    # Load RAM model
    DAPE = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path, image_size=384, vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    DAPE = DAPE.to(dtype=weight_dtype)
    return JSONResponse(content={"message": "Model loaded successfully."})


# Process input folder and save output images
@app.post("/process_folder")
def process_folder(
        input_folder: str = Form(...),
        output_folder: str = Form(...),
):
    global model, DAPE

    if not model or not DAPE:
        return JSONResponse(content={"error": "Model not loaded."}, status_code=400)

    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    os.makedirs(output_folder, exist_ok=True)

    if args.save_prompts:
        txt_folder = os.path.join(output_folder, 'txt')
        os.makedirs(txt_folder, exist_ok=True)

    for image_path in image_paths:
        input_image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = input_image.size

        # Resize input image if necessary
        resize_flag = False
        rscale = args.upscale
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # Generate captions
        lq = tensor_transforms(input_image).unsqueeze(0).to("cuda")
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, DAPE)
        prompts = f"{captions[0]}, {args.prompt},"
        validation_prompt = []
        for p in prompts.split(','):
            if p.strip() not in neg_prompt:
                validation_prompt.append(p)
        validation_prompt = ','.join(validation_prompt)

        # Save captions if required
        if args.save_prompts:
            txt_path = os.path.join(txt_folder, os.path.basename(image_path).split('.')[0] + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)

        # Generate output image
        with torch.no_grad():
            lq = lq * 2 - 1
            output_image = model(lq, prompt=validation_prompt)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize((ori_width * args.upscale, ori_height * args.upscale))

            output_pil.save(os.path.join(output_folder, os.path.basename(image_path)))

    return JSONResponse(content={"message": "Processing completed."})


# Clear model
@app.get("/clear_model")
def clear_model():
    global model, DAPE
    model = None
    DAPE = None
    torch.cuda.empty_cache()
    return JSONResponse(content={"message": "Model cleared from memory."})


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
