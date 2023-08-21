import os
from clip_interrogator import Config, Interrogator
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from clip_interrogator import Config, Interrogator
from pytorch_fid.inception import InceptionV3
import cv2

import shutil




#####   
device = 'cuda'

# load BLIP and ViT-L https://huggingface.co/openai/clip-vit-large-patch14
caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
CI_MODELS = ['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k']
CI_MODE = ['best', 'fast', 'classic', 'negative']
SAVE_OPTIONS = ['Yes', 'No']
UNLEARNING_ALGMS = ['algm1', 'algm2', 'algm3']
current_clip_model = 'ViT-H-14/laion2b_s32b_b79k'

ci_config = Config()
ci_config.clip_model_name = CI_MODELS[0]
ci_config.caption_model_name = caption_model_name
ci = Interrogator(ci_config)
ci.clip_model = ci.clip_model.to(device)


# Loading inception model for CPDM metric
def get_inception_model():
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3(range(4)).to('cuda')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - torch.mean(x)))
    ])
    return inception_model, transform


# cpdm metric function
def mean_CLIPscore(ci, origin, output):
    image_features = ci.image_to_features(origin)
    output_features = ci.image_to_features(output)
    # print(output_features.shape)
    cosine_similarity = torch.cosine_similarity(image_features, output_features, dim=-1)
    return torch.mean(cosine_similarity, dim=-1)

def mean_CLIPloss(ci, origin, output, need_cosine = False):
    image_features = ci.image_to_features(origin)
    output_features = ci.image_to_features(output)
    # print(output_features.shape)
    content_loss = torch.sum((image_features - output_features) ** 2, dim=-1)
    if need_cosine:
        cosine_similarity = torch.cosine_similarity(image_features, output_features, dim=-1)
        return torch.mean(content_loss, dim=-1).item(), torch.mean(cosine_similarity, dim=-1).item()
    return torch.mean(content_loss, dim=-1).item()
    
    
def gram_matrix(y):
    """Gram matrix."""
    
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def mean_Inceptionloss(model, origin, output, device='cuda', transform=None, weight = None, need_cosine = False):
    model.eval()

    if transform is not None:
        origin = transform(origin).unsqueeze(0).to(device)
        output = transform(output).unsqueeze(0).to(device)

    grams = []
    with torch.no_grad():
        origin_output = model(origin)
        output_output = model(output)
        
        for ori, out in zip(origin_output, output_output):
            gram_ori = gram_matrix(ori)
            gram_out = gram_matrix(out)
            grams.append(torch.sum((gram_ori - gram_out) ** 2, dim=(-2, -1)))
            # print(gram_out.shape)
    grams = torch.stack(grams)
    # print(grams.shape)
    if weight == None:
        weight = torch.ones_like(grams).to(device)
    else:
        weight = weight.unsqueeze(1).to(device)
        
    style_loss = torch.mean(weight * grams, dim=-1).mean(dim = 0).item()
    if need_cosine:
        return grams * weight, torch.mean(torch.cosine_similarity(origin_output[-1].squeeze(3).squeeze(2), output_output[-1].squeeze(3).squeeze(2), dim=-1)).item()
    
    return style_loss


def mean_Inceptionscore(model, origin, output, device='cuda', transform=None):
    model.eval()

    if transform is not None:
        origin = transform(origin).unsqueeze(0).to(device)
        output = transform(output).unsqueeze(0).to(device)

    with torch.no_grad():
        origin_is = model(origin)[-1].squeeze(3).squeeze(2)
        output_is = model(output)[-1].squeeze(3).squeeze(2)

    # print(output_is.shape)
    cosine_similarity = torch.mean(torch.cosine_similarity(origin_is, output_is, dim=-1))
    return cosine_similarity

def total_similarity(model, transform, ci, origin, output):
    weight = torch.tensor([5e-1, 1e-1, 6e4, 4e1])
    assert os.path.exists(origin)
    origin = Image.open(origin).convert('RGB')
    for i in output:
        assert os.path.exists(i)
    output = [cv2.o(i).convert('RGB') for i in output]
    mC = mean_CLIPloss(ci, origin, output, need_cosine=True)
    mI = mean_Inceptionloss(model, origin, output, transform=transform, weight=weight, need_cosine=True) 
    return mC, mI  

def similarity(model, transform, ci, origin, output):
    weight = torch.tensor([5e-1, 1e-1, 6e4, 4e1])
    assert os.path.exists(origin)
    origin = Image.open(origin).convert('RGB')
    output = Image.open(output).convert('RGB')
    
    mC = mean_CLIPloss(ci, origin, output)
    mI = mean_Inceptionloss(model, origin, output, transform=transform, weight=weight) 
    return (mC * mI) ** 2

def cpdm_metric(origin_path, outputs_path):
    
    inception_model, transform = get_inception_model()
    
    sum_loss = 0
    count = 0

    images = [str.split(file, '.')[0] for file in os.listdir(origin_path) if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')]
    origins = [os.path.join(origin_path, file) for file in os.listdir(origin_path) if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')]
    outputs = [os.path.join(outputs_path, file) for file in os.listdir(outputs_path) if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')]

    print(origins)

    metrics = {}
    
    global ci
    ci_used = ci
        
    for origin, output, image in tqdm(zip(origins, outputs, images)):
        metrics[image] = (similarity(model=inception_model, transform=transform, ci=ci_used, origin=origin, output=output))
        sum_loss += metrics[image]
        count += 1
        print(f'similarity_image{count}:', metrics[image])

    print('mean_similarity:', sum_loss / count)
    
    out_metric = sum_loss / count

    return out_metric

