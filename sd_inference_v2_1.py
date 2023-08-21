import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler 
from ldm.models.diffusion.plms import PLMSSampler
import json


torch.set_grad_enabled(False)



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    try:
        sd = pl_sd["state_dict"]
    except:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def wiki_txt2img(model,prompt,outdir,img_name, count, h, w, bs, scale=9.0):
    seed=23
    opt_device="cuda"
    plms=False
    dpm=False
    
    n_samples=bs
    n_rows=0
    n_iter=1
    from_file=None
    
    fixed_code=False
    C=4
    H=h
    W=w
    f=8
    torchscript=False
    ipex=False
    bf16=False
    scale=scale
    repeat=1
    ddim_eta=0.0
    precision="autocast"
    steps=50

    seed_everything(seed)

    
    device = torch.device("cuda") 
    model=model
    sampler = DDIMSampler(model, device=device)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(repeat)]
            data = list(chunk(data, batch_size))

    sample_count = 0
    
    base_count = (count) * 9 
    grid_count = count


    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    if torchscript or ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if bf16 else nullcontext()
        shape = [C, H // f, W // f]

        if bf16 and not torchscript and not ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
       
        if bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if ipex:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if torchscript:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                    "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 64, 64, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 768, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 64, 64, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if precision=="autocast" or bf16 else nullcontext
    with torch.no_grad(), precision_scope(opt_device):#, model.ema_scope():
        for n in trange(n_iter, desc="Sampling"):
            # for prompts in tqdm(data, desc="data"):
            for prompts in data:
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [C, H // f, W // f]
                samples, _ = sampler.sample(S=steps,
                                                conditioning=c,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(outpath, f"{img_name}_{base_count:05}.jpg"))
                    base_count += 1
                    sample_count += 1

    print("The prompt is : ",prompt)
    print("The samples are ready here: ", outpath)



# path config
ckpt_opt   ='path_to/v2-1_768-ema-pruned.ckpt'
caption_path ='path_to/caption.csv'
config_opt = 'path_to/configs/v2-v.yaml'
output_dir = 'path_to/output'
os.makedirs(output_dir, exist_ok=True)


config = OmegaConf.load(f"{config_opt}")
device = torch.device("cuda") 
model = load_model_from_config(config, f"{ckpt_opt}", device)

model.model.diffusion_model.flag_dict = {'pruning_in_modle': False, 'optim_update': False}      

count=0  
batch_size = 1

import csv
with open(caption_path, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        img_name = row['image']
        prompt = row['prompt']
        img_name_clean=img_name.split('.')[0]
        print(img_name_clean)
        
        wiki_txt2img(model=model,prompt=prompt,outdir=output_dir,img_name=img_name_clean, \
            count=count, h=768, w=768, bs=batch_size, scale=9.0)
        
        count+=1
        
        