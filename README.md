# Custom Clip Vision Preprocessor for reForge

## Introduction

Add IP-Adapter preprocessors that extend the clip vision models (h/g) to accept 448 input. This is to be used with my 
personal [Ip-Adapter finetune](https://civitai.com/models/1233692?modelVersionId=1390253) for NoobAI-XL style transfer, which is trained with guide images of size 448x448.

It re-uses the official clip vision models downloaded from ip adapters' huggingface repo and simply interpolates the positional embedding. So if you already have it locally, there should be no model downloading.

[Here](https://github.com/vahlok-alunmid/ComfyUI-ExtendIPAdapterClipVision) is the ComfyUI counterpart of this extension.

## How To Install

Go to *Extensions* tab, *Install from URL* sub tab. Copy this repo's addess to the URL textbox. Click Install. Reboot the WebUI.

You can also `git clone` or unzip the source code to the extensions folder.

## How To Use

Under the ControlNet section, choose the new preprocessor "CLIP-ViT-bigG-448 (IPAdapter)".

![Example](./example/demo.png)
