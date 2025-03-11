# Custom Clip Vision Preprocessor for Forge/reForge

## Introduction

Add IP-Adapter preprocessors that extend the clip vision models (h/g) to accept 448 input. This is to be used with my 
personal [Ip-Adapter finetune](https://civitai.com/models/1233692?modelVersionId=1390253) for NoobAI-XL style transfer, which is trained with guide images of size 448x448.

It re-uses the official clip vision models downloaded from ip adapters' huggingface repo and simply interpolates the positional embedding. So if you already have it locally, there should be no model downloading.

[Here](https://github.com/vahlok-alunmid/ComfyUI-ExtendIPAdapterClipVision) is the ComfyUI counterpart of this extension.

### ChangeLog
- 2025.3.11: Add compatibility support for Forge. Tested on Forge commit [c055f2d
](https://github.com/lllyasviel/stable-diffusion-webui-forge/commit/c055f2d43b07cbfd87ac3da4899a6d7ee52ebab9), ReForge main commit [eca3af6
](https://github.com/Panchovix/stable-diffusion-webui-reForge/commit/eca3af64f39a6a1a1a8601b7df019597d490c165), ReForge dev commit [b0487fd](https://github.com/Panchovix/stable-diffusion-webui-reForge/commit/b0487fd7b4f3ebe1c78d0932ef4ef7d0c7966d69).
- 2025.2.12: Fix for main branch of ReForge WebUI.
- 2025.2.9: Initial support for ReFroge WebUI.

## How To Install

Go to *Extensions* tab, *Install from URL* sub tab. Copy this repo's addess to the URL textbox. Click Install. Reboot the WebUI.

You can also `git clone` or unzip the source code to the extensions folder.

## How To Use

Under the ControlNet section, choose the new preprocessor "CLIP-ViT-bigG-448 (IPAdapter)".

![Example](./example/demo.png)
