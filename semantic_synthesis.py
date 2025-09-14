import os
import torch
import torchvision.transforms.functional as F

from Semantic_synthesis_util.pipeline_stable_diffusion_opt import StableDiffusionPipeline
from Semantic_synthesis_util.injection_utils import register_attention_editor_diffusers
from Semantic_synthesis_util.bounded_attention import BoundedAttention
import Semantic_synthesis_util.utils

from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything

import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, default=f"12", help='folder path')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prompt', type=str, default="a photo of <asset0>")
    parser.add_argument('--subject_token_indices', type=list, default=[[4]], help='The position of asset in prompt(start from 1), like"[[4],[6],[8]]"')
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--batch_size', type=int, default=3)


    
    args = parser.parse_args()
    return args

def load_model(device, model_id):
    scheduler = DDIMScheduler(beta_start=0.00085, 
                              beta_end=0.012, 
                              beta_schedule="scaled_linear", 
                              clip_sample=False, 
                              set_alpha_to_one=False)
    model = StableDiffusionPipeline.from_pretrained(f"Model_Ours/{model_id}", 
                                                    scheduler=scheduler, 
                                                    cross_attention_kwargs={"scale": 0.5}, 
                                                    torch_dtype=torch.float32, 
                                                    use_safetensors=True).to(device)
    return model


def run(
    boxes,
    prompt,
    subject_token_indices,
    model,
    device,
    out_dir='02',
    example_test = 0,
    seed=55,
    batch_size=3,
    filter_token_indices=None,
    eos_token_index=None,
    init_step_size=8,
    final_step_size=2,
    first_refinement_step=15,
    num_clusters_per_subject=3,
    cross_loss_scale= 1.5, # 1.5,
    self_loss_scale= 0.5, # 0.5,
    alpha_max = 0.4,
    alpha_min = 0.1,
    alpha_desc = True,
    classifier_free_guidance_scale=7.5,
    num_gd_iterations=5,
    loss_threshold=0.2,
    num_guidance_steps=15,
):


    seed_everything(seed)
    prompts = [prompt] * batch_size
    start_code = torch.randn([len(prompts), 4, 64, 64], device=device)

    editor = BoundedAttention(
        boxes,
        prompts,
        subject_token_indices,
        cross_loss_layers=list(range(12, 20)),
        self_loss_layers=list(range(12, 20)),
        cross_mask_layers=list(range(14, 20)),
        self_mask_layers=list(range(14, 20)),
        filter_token_indices=filter_token_indices,
        eos_token_index=eos_token_index,

        alpha_max=alpha_max,
        alpha_min=alpha_min,
        alpha_desc = alpha_desc,

        cross_loss_coef=cross_loss_scale,
        self_loss_coef=self_loss_scale,
        max_guidance_iter=num_guidance_steps,
        max_guidance_iter_per_step=num_gd_iterations,
        start_step_size=init_step_size,
        end_step_size=final_step_size,
        loss_stopping_value=loss_threshold,
        min_clustering_step=first_refinement_step,
        num_clusters_per_box=num_clusters_per_subject,
        debug=False,
        map_dir="logs",

        #
        use_cross_mask=True,
        use_self_mask=True,
    )

    register_attention_editor_diffusers(model, editor)
    images,prompt = model(prompts, 
                   latents=start_code, 
                   guidance_scale=classifier_free_guidance_scale,
                   device = device)

    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.join(out_dir, f"sample_{example_test}")
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))

    for i, image in enumerate(images):
        image = F.to_pil_image(image)
        image.save(os.path.join(out_dir, f'{sample_count + i}.png'))

    print("Syntheiszed images are saved in", out_dir)

def call(prompt, subject_token_indices, boxes, model, device, out_dir, object_id, batch_size = 1, seed = None):
    if seed == None:
        seed = np.random.randint(0, 9999, 1)
    run(boxes, 
        prompt, 
        subject_token_indices, 
        batch_size=batch_size,
        model = model,
        device = device,
        example_test = object_id,
        out_dir = out_dir,
        alpha_desc = True,
        seed = seed, 
        init_step_size = 8, 
        final_step_size = 2
    )
    pass

def get_boxes():
    boxes = [
        [0.25, 0.25, 0.75, 0.75]
    ]

    # boxes = [
    #     [0.05, 0.4, 0.25, 0.6],
    #     [0.35, 0.4, 0.55, 0.6],
    #     [0.65, 0.4, 0.85, 0.6],
    # ]
   
    # boxes = [
    #     [0.05, 0.35, 0.35, 0.65],
    #     [0.65, 0.35, 0.95, 0.65],
    # ]
    
    return boxes



def main():
    arg = get_args()
    
    instance_idx = arg.model_id

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device, instance_idx)
    boxes = get_boxes()

    call(arg.prompt, 
         arg.subject_token_indices, 
         boxes, 
         model, 
         device, 
         arg.output_path, 
         instance_idx, 
         batch_size= arg.batch_size,
         seed = arg.seed
        )



if __name__ == "__main__":
    main()
