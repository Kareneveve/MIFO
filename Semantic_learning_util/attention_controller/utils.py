import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
# from diffusers.models.cross_attention import AttnProcessor
from diffusers.models.attention_processor import Attention

import cv2
from typing import Union, Tuple, List, Dict, Optional

import abc

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = attn.clone()
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img

def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
) -> Image.Image:
    """Displays a list of images in a grid."""
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    return pil_img

        

## version 2.0 attention control 
'''
def register_attn_control(unet, controller, cache=None):
    def attn_forward(self, place_in_unet):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            q = self.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            inner_dim = k.shape[-1]
            head_dim = inner_dim // self.heads

            q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if is_cross and controller.cur_self_layer in controller.self_layers:
                cache.add(q, k, v, hidden_states)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(q.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            if is_cross:
                controller.cur_self_layer += 1

            return hidden_states

        return forward

    def modify_forward(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = attn_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, "children"):
                count = modify_forward(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        if "down" in net_name:
            cross_att_count += modify_forward(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += modify_forward(net, 0, "up")
        elif "up" in net_name:
            cross_att_count += modify_forward(net, 0, "mid")
    controller.num_self_layers = cross_att_count // 2
    
'''
class Controller:
    def step(self):
        self.cur_cross_layer = 0

    def __init__(self, cross_layers=(0, 16)):
        self.num_cross_layers = -1
        self.cur_cross_layer = 0
        self.cross_layers = list(range(*cross_layers))


class DataCache:
    def __init__(self):
        self.attention_map = []
        self.res_list = []

    def clear(self):
        self.attention_map.clear()
        self.res_list.clear()

    def add(self, attn_map:torch.Tensor, head):
        attn_map = attn_map.clone()
        bh, q_len, k_len = attn_map.shape
        
        res = 0
        if q_len == 8*8:
            res = 8
        elif q_len == 16*16:
            res = 16
        elif q_len == 32*32:
            res = 32
        else:
            res = 64
        
        attn_map = attn_map.reshape(-1, head, res, res, k_len)
        attn_map = attn_map.sum(dim=1) / head
        self.attention_map.append(attn_map)
        self.res_list.append(res)

    def get(self):
        return self.attention_map.copy(), self.res_list.copy()


def memory_efficient(model):
    model.freeze()
    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")

    try:
        model.enable_vae_tiling()
    except AttributeError:
        print("enable_vae_tiling is not supported.")
        
class P2PCrossAttnProcessor:
    def __init__(self, controller, cache, place_in_unet):
        super().__init__()
        self.controller = controller
        self.cache = cache
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,                      # :x from self-attention
        encoder_hidden_states=None,         # :x from text-embedding
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)    

        is_cross = encoder_hidden_states is not None  
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)   # [batch_size * heads, seq_len, dim // heads]
        key = attn.head_to_batch_dim(key)       # [batch_size * heads, seq_len, dim // heads]
        value = attn.head_to_batch_dim(value)   # [batch_size * heads, seq_len, dim // heads]

        attention_probs = attn.get_attention_scores(query, key, attention_mask)     # [batch_size * heads, q_len, k_len]

        # one line change
        if is_cross and self.controller.cur_cross_layer in self.controller.cross_layers:
            self.cache.add(attention_probs, attn.heads)
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if is_cross:
            self.controller.cur_cross_layer += 1

        return hidden_states