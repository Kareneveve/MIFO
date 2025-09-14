import os
import warnings
import hashlib
import itertools
from pathlib import Path
from typing import List, Optional
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig, AutoTokenizer
from diffusers import StableDiffusionDepth2ImgPipeline, DiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler, DDIMScheduler

from lora_diffusion import (
            extract_lora_ups_down,
            inject_trainable_lora,
            safetensors_available,
            save_lora_weight,
            save_safeloras,
        )
# from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers

from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from PIL import Image
from tqdm.auto import tqdm

from .utils import AttentionStore, Controller, DataCache, P2PCrossAttnProcessor, text_under_image, view_images

def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

class ExtractorDataset(Dataset):
    def __init__(
        self,
        instance_data_path,
        instance_mask_paths:List,
        object_name,
        placeholder_tokens,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        num_of_assets=1,
        use_trashbin=False,
        use_union_sample=True,
        flip_p=0.5,
    ):
        self.object_name = object_name
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        
        self.use_trashbin = use_trashbin
        self.use_union_sample = use_union_sample
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [   
                transforms.Resize([size, size]),
                transforms.ToTensor(),
            ]
        )

        self.instance_data_root = Path(instance_data_path)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        if self.use_trashbin:
            self.trash_token = placeholder_tokens[-1]
            self.placeholder_tokens = placeholder_tokens[:-1]
        else:
            self.placeholder_tokens = placeholder_tokens

        # one instance view
        instance_img = Image.open(instance_data_path)
        self.instance_image = self.image_transforms(instance_img)

        # one instance view
        self.instance_masks = []
        for instance_mask_path in instance_mask_paths:
            curr_mask = Image.open(instance_mask_path)
            curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
            self.instance_masks.append(curr_mask)
        
        self.instance_masks = torch.cat(self.instance_masks)

        self._length = 1
        
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # Marked by Hao
        num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
        if self.use_union_sample == False:
            num_of_tokens = 1
        
        tokens_ids_to_use = random.sample(
            range(len(self.placeholder_tokens)), k=num_of_tokens
        )
        
        tokens_to_use = [self.placeholder_tokens[tkn_i]
                          for tkn_i in tokens_ids_to_use]
        prompt = f"a photo of " + " and ".join(tokens_to_use)
        
        if self.use_trashbin:
            prompt += (" " + self.trash_token)

        # modify by Hao, Mask real image
        example["instance_images"] = self.instance_image # * self.instance_masks[tokens_ids_to_use].squeeze(0)
        example["instance_masks"] = self.instance_masks[tokens_ids_to_use]
        example["token_ids"] = torch.tensor(tokens_ids_to_use)

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
                
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]

    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["class_images"] for example in examples] + pixel_values

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    masks = torch.stack(masks)
    token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
        "token_ids": token_ids,
    }
    return batch

class TextExtractor():
    def __init__(self,
                 pretrained_model_name_or_path,
                 instance_data_path,
                 instance_mask_path,
                 object_name,
                 class_data_dir,
                 class_prompt,
                 num_of_assets,
                 with_prior_preservation=False,
                 prior_loss_weight=1.0,
                 num_class_images=100,
                 output_dir="output",
                 seed=None,
                 resolution=512,
                 center_crop=False,
                 train_text_encoder=True,
                 train_steps=400,
                 reward_steps=0,
                 DB_training_steps=0,
                 learning_rate=2e-6,
                 initial_learning_rate=5e-2,
                 DB_lr=2e-6,
                 scale_lr=False,
                 lr_scheduler="cosine_with_restarts",
                 # lr_scheduler="constant",
                 mixed_precision="fp16",
                 lambda_attention=1e-2,
                 initializer_tokens=[],
                 trash_initializer_token=None,
                 placeholder_token="<asset>",
                 apply_masked_loss=True,
                 log_checkpoints=True,
                 log_steps=50,
                 attn_layers=[6, 10],
                 extract_layers = 900,
                 reg_weight_reward = 1e-2,
                 reward_alpha = 0.2,
                 reg_weight_punish = 1e-2,
                 rec_weight = 1,
                 sim_bas = False,
                 use_union_sample = True,
                 
                 trash_weight = 1e-4,
                 use_sd_loss = True,
                 use_trashbin = False,
                 need_DB = False,
                 ): 
        assert len(initializer_tokens) == 0 or len(initializer_tokens) == num_of_assets
        if with_prior_preservation:
            if class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
        else:
            # logger is not available yet
            if class_data_dir is not None:
                warnings.warn(
                    "You need not use --class_data_dir without --with_prior_preservation."
                )
            if class_prompt is not None:
                warnings.warn(
                    "You need not use --class_prompt without --with_prior_preservation."
                )
        
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.output_dir = output_dir
        self.class_prompt = class_prompt
        self.class_data_dir = class_data_dir
        self.object_name = object_name
        self.instance_mask_path = instance_mask_path
        self.instance_data_path = instance_data_path
        self.prior_loss_weight = prior_loss_weight        
        self.train_text_encoder = train_text_encoder
        self.lambda_attention = lambda_attention
        self.apply_masked_loss = apply_masked_loss
        self.with_prior_preservation = with_prior_preservation
        self.train_batch_size = 1
        self.num_of_assets = num_of_assets
        self.log_steps = log_steps
        self.attn_layers = attn_layers
        self.extract_layers = extract_layers
        self.reg_weight_reward = reg_weight_reward
        self.reward_alpha = reward_alpha
        self.reg_weight_punish = reg_weight_punish
        self.rec_weight = rec_weight
        self.use_sd_loss = use_sd_loss
        self.use_trashbin = use_trashbin 
        self.trash_weight = trash_weight
        self.sim_bas = sim_bas
        self.use_union_sample = use_union_sample
        
        self.reward_steps = reward_steps
        self.DB_training_steps = DB_training_steps
        self.need_DB = need_DB
        self.DB_lr = DB_lr
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=mixed_precision,
        )
        
        if seed is not None:
            set_seed(seed)
        
        # Generate class images if prior preservation is enabled.
        if with_prior_preservation:
            self.generate_class_image(class_data_dir, num_class_images, pretrained_model_name_or_path, class_prompt, prior_generation_precision="fp32")
        
        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
        # import correct text encoder class
        text_encoder_cls =  CLIPTextModel
        
        # Load scheduler and models
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler.set_timesteps(1000)
        
        self.infer_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=False,
            )
        
        # Add assets tokens to tokenizer
        self.placeholder_tokens = [
            placeholder_token.replace(">", f"{idx}>")
            for idx in range(self.num_of_assets)
        ]
        
        if self.use_trashbin:
            self.placeholder_tokens += ["<trashbin>"]
            initializer_tokens += [trash_initializer_token]
            self.num_of_assets += 1
        
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        
        assert num_added_tokens == self.num_of_assets
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens
        )
        if self.use_trashbin:
            self.trash_token_ids = self.placeholder_token_ids[-1]
        
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # self.instance_prompt = "a photo of chair with " + self.placeholder_tokens[0] + " material, " + self.placeholder_tokens[1] + " material"
        
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        
        # replace new token with initial_token
        if len(initializer_tokens) != 0:
            for tkn_idx, initializer_token in enumerate(initializer_tokens):
                curr_token_ids = self.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                # assert (len(curr_token_ids)) == 1
                token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
                    curr_token_ids[0]
                ]
        
        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        
        optimizer_class = torch.optim.AdamW
        
        # We start by only optimizing the embeddings
        params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=initial_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-2,
            eps=1e-8,
        )
        
        # Dataset and DataLoaders creation:
        self.train_steps = train_steps
        self.train_dataset = ExtractorDataset(
            instance_data_path=instance_data_path,
            instance_mask_paths=instance_mask_path,
            object_name=object_name,
            placeholder_tokens=self.placeholder_tokens,
            class_data_root=class_data_dir
            if with_prior_preservation
            else None,
            class_prompt=class_prompt,
            tokenizer=self.tokenizer,
            size=resolution,
            center_crop=center_crop,
            num_of_assets=num_of_assets,
            use_trashbin=self.use_trashbin,
            use_union_sample = use_union_sample,
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, with_prior_preservation
            ),
            num_workers=1,
        )
        
        self.lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=10,
            num_training_steps=train_steps,
        )
        
        (
            self.unet,
            self.text_encoder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        
        self.weight_dtype = torch.float32
        
        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        self.num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_train_epochs = train_steps // self.num_update_steps_per_epoch
    
    def train(self):
        total_batch_size = 1
        global_step = 0
        first_epoch = 0
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        
        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )
        
        # Create attention controller
        # self.controller = AttentionStore()
        self.controller = Controller()
        self.cache = DataCache()
        self.register_attention_control(self.controller, self.cache)
        
        self.time_sample_control = 0
        
        for epoch in range(first_epoch, self.num_train_epochs):
            if self.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                logs = {}
                with self.accelerator.accumulate(self.unet):
                    # get merged mask
                    max_masks = torch.max(
                        batch["instance_masks"], axis=1
                    ).values
                    
                    # prepare images 
                    input_image = batch["pixel_values"].to(dtype=self.weight_dtype)
                    
                    if self.output_dir.split('/')[0] == "Model_TI" or self.use_union_sample == False:
                        input_image[0] = input_image[0] * max_masks[0]
                    
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    # Sample noise thta we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample a random timestep for each image
                    # 去随机化处理
                    if self.time_sample_control == 0:
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            # self.extract_layers,
                            (bsz,),
                            device=latents.device,
                        )
                        self.time_sample_control = 1
                    else:
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            # self.extract_layers,
                            (bsz,),
                            device=latents.device,
                        )
                        self.time_sample_control = 0
                        
                    timesteps = timesteps.long()
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    
                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    
                    # Predict the noise residual
                    latent_model_input = noisy_latents
                    
                    model_pred = self.unet(
                        latent_model_input, timesteps, encoder_hidden_states
                    ).sample
                    
                    target = noise
                    
                    attn_maps, res_list = self.cache.get()
                    
                    loss = 0
                    if self.use_sd_loss:
                        if self.with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                            target_prior, target = torch.chunk(target, 2, dim=0)

                            if self.apply_masked_loss:
                                max_masks = torch.max(
                                    batch["instance_masks"], axis=1
                                ).values
                                downsampled_mask = F.interpolate(
                                    input=max_masks, size=(64, 64)
                                )
                                model_pred = model_pred * downsampled_mask
                                target = target * downsampled_mask

                            # Compute instance loss
                            loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="mean"
                            )
                            

                            # Compute prior loss
                            prior_loss = F.mse_loss(
                                model_pred_prior.float(),
                                target_prior.float(),
                                reduction="mean",
                            )

                            # Add the prior loss to the instance loss.
                            loss = loss + self.prior_loss_weight * prior_loss
                        else:
                            if self.apply_masked_loss:
                                max_masks = torch.max(
                                    batch["instance_masks"], axis=1
                                ).values
                                downsampled_mask = F.interpolate(
                                    input=max_masks, size=(64, 64)
                                )
                                model_pred = model_pred * downsampled_mask
                                target = target * downsampled_mask
                                
                            loss = self.rec_weight * F.mse_loss(
                                model_pred.float(), target.float(), reduction="mean"
                            )

                    # ### ======================== 【正交】 ==================
                    # orth_loss_tensor, mean_off = self.compute_orth_loss(device=self.accelerator.device)
                    # # logs["orth_raw"] = orth_loss_tensor.detach().item() if isinstance(orth_loss_tensor, torch.Tensor) else float(orth_loss_tensor)
                    # # logs["orth_mean_off"] = mean_off
                    # loss = loss + 0.001 * orth_loss_tensor

                    # ### ======================== 【正交】 ==================
                            
                    reg_loss = 0.0
                    trash_loss = 0.0
                    curr_cond_batch_idx = 0 + self.train_batch_size if self.with_prior_preservation else 0
                    for layer in self.attn_layers: 
                        # prepare mask and attn_map
                        attn_map = attn_maps[layer][curr_cond_batch_idx].unsqueeze(0)
                        res = res_list[layer]
                        GT_mask = F.interpolate(
                        input=batch["instance_masks"][0], size=(res, res), mode="bilinear"
                        )
                        zero_mask = torch.zeros_like(GT_mask[0])
                        
                        # Attention loss 
                        if self.reg_weight_punish != 0 and timesteps[0] < self.extract_layers:
                            for mask_id in range(len(batch["instance_masks"][0])):
                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    batch["token_ids"][0][mask_id]
                                ]
                                
                                asset_idx = (
                                    (
                                        batch["input_ids"][curr_cond_batch_idx]
                                        == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                # award loss
                                if global_step < self.reward_steps:
                                    gt_mask = GT_mask[mask_id] * self.reward_alpha
                                    
                                    asset_attn_mask = attn_map[..., asset_idx]
                                    if self.sim_bas:
                                        asset_attn_mask = asset_attn_mask / asset_attn_mask.max()
                                    
                                    reg_loss += self.reg_weight_reward * F.mse_loss(
                                            gt_mask.float(),
                                            asset_attn_mask.float(),
                                            reduction="mean",
                                    )
                                # punish loss
                                else:
                                    neg_gt_mask = 1 - GT_mask[mask_id]
                                    asset_attn_mask = attn_map[..., asset_idx]
                                    reg_attn_mask = asset_attn_mask * neg_gt_mask
                                    
                                    reg_loss += self.reg_weight_punish * F.mse_loss(
                                            zero_mask.float(),
                                            reg_attn_mask.float(),
                                            reduction="mean",
                                    )
                        # trash loss
                        if self.use_trashbin and timesteps[0] < self.extract_layers:
                            trash_asset_idx = (
                                (
                                    batch["input_ids"][0]
                                    == self.trash_token_ids
                                )
                                .nonzero()
                                .item()
                            )
                            
                            trash_attn_map = attn_maps[layer][0].unsqueeze(0)
                            trash_asset_attn_mask = trash_attn_map[..., trash_asset_idx]
                            
                            trash_loss += F.mse_loss(
                                zero_mask.float(),
                                trash_asset_attn_mask.float(),
                                reduction="mean",
                            )
                                
                    
                    logs["reg_loss"] = reg_loss if isinstance(reg_loss, float) else reg_loss.detach().item()
                    loss += reg_loss
                    
                    trash_loss = self.trash_weight * trash_loss
                    logs["trash_loss"] = trash_loss if isinstance(trash_loss, float) else trash_loss.detach().item()
                    loss += trash_loss
                        
                    self.accelerator.backward(loss)
                    
                    params_to_clip = (
                        itertools.chain(
                            self.unet.parameters(), self.text_encoder.parameters()
                        )
                        if self.train_text_encoder
                        else self.unet.parameters()
                    )
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, 1.0
                    )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=False)
                    
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    with torch.no_grad():
                        self.accelerator.unwrap_model(
                            self.text_encoder
                        ).get_input_embeddings().weight[
                            : -self.num_of_assets
                        ] = orig_embeds_params[
                            : -self.num_of_assets
                        ]
                    
                progress_bar.update(1)
                global_step += 1
                
                if global_step % self.log_steps == 0:
                    attn_maps, _ = self.cache.get()
                    self.show_attention_map(global_step, timesteps[0].item(), batch, attention_maps=attn_maps, log_root="logs")
                
                # No need to keep the attention store
                self.controller.step()
                self.cache.clear()
                
                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.train_steps:
                    break
        
        self.save_pipeline(self.output_dir)
        self.accelerator.end_training()
        
    def DreamBooth(self, out_model_dir=None, load_path=None, lora_out_dir=None, train_text_embeddings=True, use_union_sample=False):
        assert lora_out_dir is not None
        self.lora_out_dir = lora_out_dir
        
        # set presevation
        self.with_prior_preservation = True
        self.attn_layers = [4, 5, 7, 8, 9]
        self.train_text_embeddings = train_text_embeddings
        
        # load text_encoder and Tokenizer
        if load_path is not None:
            self.text_encoder = CLIPTextModel.from_pretrained(
                load_path,
                subfolder="text_encoder",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    subfolder="tokenizer",
                    use_fast=False,
            )
        
        # modify dataloader's attribute
        self.train_dataset = ExtractorDataset(
            instance_data_path=self.instance_data_path,
            instance_mask_paths=self.instance_mask_path,
            object_name=self.object_name,
            placeholder_tokens=self.placeholder_tokens,
            class_data_root=self.class_data_dir,
            class_prompt=self.class_prompt,
            tokenizer=self.tokenizer, # modify
            num_of_assets=self.num_of_assets,
            use_trashbin=self.use_trashbin,
            use_union_sample=use_union_sample,
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.with_prior_preservation
            ),
            num_workers=1,
        )
        
        self.freeze_all_model()
        
        # reset model's requires_grad and optimizer
        del self.optimizer
        del self.lr_scheduler
        
        self.unet.requires_grad_(True)
        self.text_encoder.requires_grad_(True)
        
        unet_params = self.unet.parameters()
        
        params_to_optimize = (
            itertools.chain(
                unet_params, 
                self.text_encoder.parameters())
            if self.train_text_embeddings
            else itertools.chain(unet_params,
                                 self.text_encoder.get_input_embeddings().parameters())
        )
        
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
        
        self.optimizer = optimizer_class(
                            params_to_optimize,
                            lr=self.DB_lr,
                            betas=(0.9, 0.99),
                            weight_decay=1e-2,
                            eps=1e-8,
                        )
        
        self.lr_scheduler = get_scheduler(
                    "constant",
                    optimizer=self.optimizer,
                    num_training_steps=self.DB_training_steps,
                )
        
        self.text_encoder, self.tokenizer, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            self.text_encoder, self.tokenizer, self.optimizer, self.lr_scheduler, self.train_dataloader
        )
        
        self.unet.train()
        if self.train_text_embeddings:
            self.text_encoder.train()
        
        global_step = 0
        progress_bar = tqdm(
            range(global_step, self.DB_training_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        
        
        self.num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_train_epochs = max(self.DB_training_steps // self.num_update_steps_per_epoch, 1)
        
        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )
        
        self.controller = Controller()
        self.cache = DataCache()
        self.register_attention_control(self.controller, self.cache)
        
        self.time_sample_control = 0
        
        for epoch in range(0, self.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                logs = {}
                with self.accelerator.accumulate(self.unet):
                    # get merged mask
                    max_masks = torch.max(
                        batch["instance_masks"], axis=1
                    ).values
                    
                    # prepare images 
                    input_image = batch["pixel_values"].to(dtype=self.weight_dtype)
                    if out_model_dir == "Model_DB" or use_union_sample == False:
                        input_image[1] = input_image[1] * max_masks[0]
                    
                    # Convert images to latent space
                    latents = self.vae.encode(
                        input_image
                    ).latent_dist.sample()
                    latents = latents * 0.18215
                    
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        # self.extract_layers,
                        (bsz,),
                        device=latents.device,
                    )
                    self.time_sample_control = 1
                    
        
                    timesteps = timesteps.long()
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )
                    
                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    
                    # Predict the noise residual
                    latent_model_input = noisy_latents
                    
                    model_pred = self.unet(
                        latent_model_input, timesteps, encoder_hidden_states
                    ).sample
                    
                    target = noise
                    
                    model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                    target_prior, target = torch.chunk(target, 2, dim=0)
                    
                    attn_maps, res_list = self.cache.get()
                    
                    if out_model_dir != "Model_DB":
                        downsampled_mask = F.interpolate(
                            input=max_masks, size=(64, 64)
                        )
                        model_pred = model_pred * downsampled_mask
                        target = target * downsampled_mask
                    
                    # Compute instance loss
                    loss =  F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                    
                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(),
                        target_prior.float(),
                        reduction="mean",
                    )
                    
                    loss = loss + prior_loss
                    
                    # text_embeddings
                    reg_loss = 0
                    curr_cond_batch_idx = 0 + self.train_batch_size if self.with_prior_preservation else 0
                    for layer in self.attn_layers: 
                        # prepare mask and attn_map
                        attn_map = attn_maps[layer][curr_cond_batch_idx].unsqueeze(0)
                        res = res_list[layer]
                        GT_mask = F.interpolate(
                        input=batch["instance_masks"][0], size=(res, res), mode="bilinear"
                        )
                        zero_mask = torch.zeros_like(GT_mask[0])
                        
                        if self.train_text_embeddings and timesteps[0] < self.extract_layers:
                            for mask_id in range(len(batch["instance_masks"][0])):
                                    curr_placeholder_token_id = self.placeholder_token_ids[
                                        batch["token_ids"][0][mask_id]
                                    ]
                                    
                                    asset_idx = (
                                        (
                                            batch["input_ids"][curr_cond_batch_idx]
                                            == curr_placeholder_token_id
                                        )
                                        .nonzero()
                                        .item()
                                    )
                                    
                                    ## with punish loss
                                    neg_gt_mask = 1 - GT_mask[mask_id]
                                    asset_attn_mask = attn_map[..., asset_idx]
                                    reg_attn_mask = asset_attn_mask * neg_gt_mask
                                    
                                    reg_loss += F.mse_loss(
                                            zero_mask.float(),
                                            reg_attn_mask.float(),
                                            reduction="mean",
                                    )
                    
                    reg_loss = self.reg_weight_punish * reg_loss
                    # reg_loss = 0.0 * reg_loss
                    loss += reg_loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.unet.parameters(), self.text_encoder.parameters()
                            )
                            if self.train_text_embeddings
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, 1.0
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    with torch.no_grad():
                        self.accelerator.unwrap_model(
                            self.text_encoder
                        ).get_input_embeddings().weight[
                            : -self.num_of_assets
                        ] = orig_embeds_params[
                            : -self.num_of_assets
                        ]
                
                progress_bar.update(1)
                global_step += 1
                
                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
        
        
                # No need to keep the attention store
                self.controller.step()
                self.cache.clear()
                
                if global_step >= self.DB_training_steps:
                    break
            
        self.save_pipeline(out_model_dir, save_lora="DB")
        self.accelerator.end_training()
        
    def step_lr(self, global_steps:int=None):
        if global_steps < 50:
            return 5e-2
        elif global_steps < 200:
            return 1e-2
        else:
            return 5e-4
        
    def inference(self, num_inference_steps=50):
        self.infer_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.infer_scheduler.timesteps.to(self.accelerator.device)
        
        instance_img = self.train_dataset.instance_image.to(self.accelerator.device).unsqueeze(0)
        latent_img = self.vae.encode(instance_img.to(dtype=self.weight_dtype)).latent_dist.sample()
        latent_img = latent_img * 0.18215
        
        prompt = ""
        instance_prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        
        encoder_hidden_states = self.text_encoder(instance_prompt_ids)[0].detach().requires_grad_(False)
        
        for i in range(num_inference_steps):
            t = timesteps[i]
            noise = torch.randn_like(latent_img)
            noise_img = self.infer_scheduler.add_noise(latent_img, noise, t)
            pred_noise = self.unet(noise_img, t, encoder_hidden_states).sample
            z0 = self.infer_scheduler.step(noise, t, noise_img).pred_original_sample
            z0 = z0.to(noise_img.dtype)
            # decode noise_img and z0
            xt = self.decode_latents(noise_img)
            xt = self.numpy_to_pil(xt)
            
            x0 = self.decode_latents(z0)
            x0 = self.numpy_to_pil(x0)
            
            xt[0].save(f"inference_log/xt_t{t}.jpg")
            x0[0].save(f"inference_log/x0_t{t}.jpg")
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = latents.to(dtype=torch.float32)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def save_pipeline(self, path, save_lora="PT"):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if save_lora == "LORA":
                path=os.path.join(path, self.lora_out_dir)
                os.makedirs(path, exist_ok=True)
                save_lora_weight(self.unet, path=os.path.join(path, "lora.pt")) 
            
            
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    # unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    tokenizer=self.tokenizer,
                    safety_checker=None,
                )
            elif save_lora == "DB":
                path=os.path.join(path, self.lora_out_dir)
                os.makedirs(path, exist_ok=True)
                
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    tokenizer=self.tokenizer,
                    safety_checker=None,
                )
            else:
                os.makedirs(path, exist_ok=True)
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    tokenizer=self.tokenizer,
                    safety_checker=None,
                )
            
            pipeline.save_pretrained(path)     
                
            
    def freeze_all_model(self):
        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    def generate_class_image(self,
                             class_data_dir,
                             num_class_images,
                             pretrained_model_name_or_path,
                             class_prompt,
                             prior_generation_precision="fp32",
                             ):
        class_images_dir = Path(class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            torch_dtype = (
                torch.float32
            )
            if prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            
            print(f"Load model torch_dtype: {torch_dtype}.")
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images

            sample_dataset = PromptDataset(class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=1
            )

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not self.accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def register_attention_control(self, controller, cache):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, cache=cache, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_cross_layers = cross_att_count // 2

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        1, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out
    
    def show_attention_map(self, global_step, t, batch, attention_maps, log_root="",):
        os.makedirs(log_root, exist_ok=True)
        # last_sentence = self.instance_prompt
        train_data_id = 1 if self.with_prior_preservation else 0
        last_sentence = batch["input_ids"][train_data_id]
        last_sentence = last_sentence[
            (last_sentence != 0)
            & (last_sentence != 49406)
            & (last_sentence != 49407)
        ]
        last_sentence = self.tokenizer.decode(last_sentence)
        for i in range(0, self.controller.num_cross_layers, 1):
            attention_map = attention_maps[i][train_data_id].detach().cpu()
            self.save_cross_attention_vis(last_sentence, attention_map, os.path.join(
                                        log_root, f"{global_step:05}_step_t{t}_layer{i}_attn.jpg"
                                    ),)
        
        # self.controller.step()
        # self.controller.attention_store = {}
        
    
    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            # image = 255 * image / image.max()
            image = 255 * image
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = view_images(np.stack(images, axis=0))
        vis.save(path)    
    
    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def compute_orth_loss(self, device=None):
        """
        用 embedding 中最后 self.num_of_assets 个 token 计算正交损失。
        返回 (orth_loss_tensor, mean_off_diag_abs)
        """
        # 快速关闭路径
        if getattr(self, "orth_loss_weight", 0.0) == 0.0:
            return torch.tensor(0.0, device=device or self.accelerator.device), 0.0

        emb = self.text_encoder.get_input_embeddings().weight  # (vocab_size, dim)
        device = device or emb.device

        vocab_size = emb.size(0)
        k = min(getattr(self, "num_of_assets", 0), vocab_size)
        if k <= 1:
            return torch.tensor(0.0, device=device), 0.0

        # 取最后 k 个 token 的 embedding
        E = emb[vocab_size - k : vocab_size].to(device)  # (k, dim)

        # L2 归一化每一行后计算 Gram 矩阵的非对角项平方和
        E_norm = F.normalize(E, p=2, dim=1)
        G = torch.matmul(E_norm, E_norm.t())  # (k, k)
        eye = torch.eye(k, device=G.device)
        off = G * (1.0 - eye)  # 把对角线置0

        orth_loss = (off * off).sum() / (k * (k - 1) + 1e-12)  # 平均化
        mean_off_abs = off.abs().sum().item() / (k * (k - 1) + 1e-12)

        return orth_loss, mean_off_abs
