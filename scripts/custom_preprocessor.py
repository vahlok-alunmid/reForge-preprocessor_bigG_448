import logging
import math
import os

import torch

from ldm_patched.modules.clip_vision import convert_to_transformers, ClipVisionModel
from ldm_patched.modules.utils import load_torch_file
from modules.util import load_file_from_url
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.supported_preprocessor import PreprocessorClipVision
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    pos_embedding: torch.Tensor,
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> torch.Tensor:
    """(From torchvision) This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        pos_embedding (torch.Tensor): Positional embedding tensor.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = torch.nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)
    else:
        new_pos_embedding = pos_embedding

    return new_pos_embedding


def load_448_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g_448.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h_448.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        if sd["vision_model.embeddings.position_embedding.weight"].shape[0] == 577:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336.json")
        else:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_448.json")
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip


def extend_clipvision_input_size(state_dict, target_size=448):
    square_patch_count = state_dict['vision_model.embeddings.patch_embedding.weight'].shape[2]
    original_dtype = state_dict['vision_model.embeddings.patch_embedding.weight'].dtype
    pos_embedding = torch.unsqueeze(
        state_dict['vision_model.embeddings.position_embedding.weight'], 0).to(torch.float32)
    new_seq_len = (target_size // square_patch_count) ** 2 + 1
    pos_embedding = interpolate_embeddings(target_size, square_patch_count, pos_embedding)
    state_dict['vision_model.embeddings.position_embedding.weight'] = torch.squeeze(pos_embedding).to(original_dtype)
    state_dict['vision_model.embeddings.position_ids'] = torch.tensor([_ for _ in range(new_seq_len)],
                                                                      dtype=torch.int64)
    return state_dict


class CustomPreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.sorting_priority = 20

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            unfold_batch=False,
        )
        return cond

    def load_clipvision(self):
        if self.clipvision is not None:
            return self.clipvision

        ckpt_path = load_file_from_url(
            url=self.url,
            model_dir=preprocessor_dir,
            file_name=self.filename
        )
        ckpt_hash = ckpt_path + '.448'
        if ckpt_hash in PreprocessorClipVision.global_cache:
            self.clipvision = PreprocessorClipVision.global_cache[ckpt_hash]
        else:
            sd = load_torch_file(ckpt_path)
            sd = extend_clipvision_input_size(sd, 448)
            if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
                self.clipvision = load_448_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
            else:
                self.clipvision = load_448_clipvision_from_sd(sd)
            PreprocessorClipVision.global_cache[ckpt_hash] = self.clipvision

        # Set up the model patcher for the CLIP vision model
        self.setup_model_patcher(self.clipvision.model)

        return self.clipvision


add_supported_preprocessor(CustomPreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-bigG-448 (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))

add_supported_preprocessor(CustomPreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-H-448 (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))
