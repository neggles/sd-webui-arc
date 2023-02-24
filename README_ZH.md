# 用于Intel Arc的Stable Diffusion网页用户界面，带有Pytorch的Intel扩展。

这个版本有点小毛病，如果你是Windows用户，可以试试DirectML版本[在这里](https://github.com/Aloereed/stable-diffusion-webui-arc-directml)或[这里](https://github.com/lshqqytiger/stable-diffusion-webui-directml)。 [英文自述文件戳这里](README.md)  

## 要求
+ Linux 64 bit / WSL2
+ 你应该按照[Intel 安装指南](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html)来完成IPEX的安装。或者查阅[ArcNotes.txt](ArcNotes.txt)。
+ Intel Arc A770/A750
+ Python 3.10

## 安装英特尔软件包的简单指南
### 0. 升级WSLg
如果你使用的是WSL2，那么你首先需要在powershell中运行 
```powershell
wsl --update
```
### 1. 安装英特尔® oneAPI基础工具包
您只需要安装英特尔® oneAPI DPC++编译器（DPCPP_ROOT为其安装路径）。
以及英特尔® oneAPI数学内核库（oneMKL）（MKL_ROOT为其安装路径）。
```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19079/l_BaseKit_p_2023.0.0.25537.sh
sudo sh ./l_BaseKit_p_2023.0.0.25537.sh
```
 默认的安装位置{ONEAPI_ROOT}对于root账户是/opt/intel/oneapi，其他账户是${HOME}/intel/oneapi。一般来说，DPCPP_ROOT是{ONEAPI_ROOT}/compiler/latest，MKL_ROOT是{ONEAPI_ROOT}/mkl/latest。  
 每次启动前都需要：
```bash
source /opt/intel/oneapi/setvars.sh
```
### 2. 安装运行时软件包
(一些用户报告说，这些软件包在apt仓库中太旧了，所以请使用手动安装。)  
~~sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero intel-media-va-driver-non-free libmfx1~~  
或者你可以这样做。
```bash
mkdir neo
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.12504.5/intel-igc-core_1.0.12504.5_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.12504.5/intel-igc-opencl_1.0.12504.5_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/22.43.24595.30/intel-level-zero-gpu-dbgsym_1.3.24595.30_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/22.43.24595.30/intel-level-zero-gpu_1.3.24595.30_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/22.43.24595.30/intel-opencl-icd-dbgsym_22.43.24595.30_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/22.43.24595.30/intel-opencl-icd_22.43.24595.30_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/22.43.24595.30/libigdgmm12_22.3.0_amd64.deb
sudo dpkg -i *.deb
cd ..
```

### 3. 用`sycl-ls`验证GPU的可见性
```bash
sycl-ls

[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2022.15.12.0.01_081451]。
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Core(TM) i5-9600KF CPU @ 3.70GHz 3.0 [2022.15.12.0.01_081451] 。
[opencl:gpu:2] Intel(R) OpenCL HD Graphics, Intel(R) Graphics [0x56a0] 3.0 [22.43.24595.30] 。
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Graphics [0x56a0] 1.3 [1.3.24595] <----- 检查这个是否存在。
```


## 设置
+ 只要运行[webui.sh](webui.sh)
+ 请验证您在运行上一步时是否安装了Pytorch的Intel版本（而不是+cu117）。
+ 把你的模型放在/home/<你的用户名>/stable-diffusion-webui/models/Stable-diffusion中。
+ **一旦配置好，你应该停止webui.sh并复制&粘贴这个资源库到/home/<你的用户名>/stable-diffusion-webui，覆盖其内容。**（或者你可以修改[webui.sh](webui.sh)中的安装位置。另外，你也可以在一切开始之前，直接把这个软件库放到那个位置。）  
+ 再次运行 [webui.sh](webui.sh) 并享受它!

## 测试
在Intel Arc A770 16GB上测试25秒，使用anything-v4.0的150步。

## 已知问题
+ 当出现黑屏或其他现象时，请在[webui-user.sh](webui-user.sh)中加入： --skip-torch-cuda-test --disable-nan-check  
+ 或者就多试试。
+ 正面提示请大于75字。
+ 推理步骤的数量要大于50步，甚至更多。

## 原有版本的Readme
A browser interface based on Gradio library for Stable Diffusion.

![](screenshot.png)

## Features
[Detailed feature showcase with images](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Color Sketch
- Prompt Matrix
- Stable Diffusion Upscale
- Attention, specify parts of text that the model should pay more attention to
    - a man in a ((tuxedo)) - will pay more attention to tuxedo
    - a man in a (tuxedo:1.21) - alternative syntax
    - select text and press ctrl+up or ctrl+down to automatically adjust attention to selected text (code contributed by anonymous user)
- Loopback, run img2img processing multiple times
- X/Y/Z plot, a way to draw a 3 dimensional plot of images with different parameters
- Textual Inversion
    - have as many embeddings as you want and use any names you like for them
    - use multiple embeddings with different numbers of vectors per token
    - works with half precision floating point numbers
    - train embeddings on 8GB (also reports of 6GB working)
- Extras tab with:
    - GFPGAN, neural network that fixes faces
    - CodeFormer, face restoration tool as an alternative to GFPGAN
    - RealESRGAN, neural network upscaler
    - ESRGAN, neural network upscaler with a lot of third party models
    - SwinIR and Swin2SR([see here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/2092)), neural network upscalers
    - LDSR, Latent diffusion super resolution upscaling
- Resizing aspect ratio options
- Sampling method selection
    - Adjust sampler eta values (noise multiplier)
    - More advanced noise setting options
- Interrupt processing at any time
- 4GB video card support (also reports of 2GB working)
- Correct seeds for batches
- Live prompt token length validation
- Generation parameters
     - parameters you used to generate images are saved with that image
     - in PNG chunks for PNG, in EXIF for JPEG
     - can drag the image to PNG info tab to restore generation parameters and automatically copy them into UI
     - can be disabled in settings
     - drag and drop an image/text-parameters to promptbox
- Read Generation Parameters Button, loads parameters in promptbox to UI
- Settings page
- Running arbitrary python code from UI (must run with --allow-code to enable)
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Tiling support, a checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
    - Can use a separate neural network to produce previews with almost none VRAM or compute requirement
- Negative prompt, an extra text field that allows you to list what you don't want to see in generated image
- Styles, a way to save part of prompt and easily apply them via dropdown later
- Variations, a way to generate same image but with tiny differences
- Seed resizing, a way to generate same image but at slightly different resolution
- CLIP interrogator, a button that tries to guess prompt from an image
- Prompt Editing, a way to change prompt mid-generation, say to start making a watermelon and switch to anime girl midway
- Batch Processing, process a group of files using img2img
- Img2img Alternative, reverse Euler method of cross attention control
- Highres Fix, a convenience option to produce high resolution pictures in one click without usual distortions
- Reloading checkpoints on the fly
- Checkpoint Merger, a tab that allows you to merge up to 3 checkpoints into one
- [Custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) with many extensions from community
- [Composable-Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), a way to use multiple prompts at once
     - separate prompts using uppercase `AND`
     - also supports weights for prompts: `a cat :1.2 AND a dog AND a penguin :2.2`
- No token limit for prompts (original stable diffusion lets you use up to 75 tokens)
- DeepDanbooru integration, creates danbooru style tags for anime prompts
- [xformers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), major speed increase for select cards: (add --xformers to commandline args)
- via extension: [History tab](https://github.com/yfszzx/stable-diffusion-webui-images-browser): view, direct and delete images conveniently within the UI
- Generate forever option
- Training tab
     - hypernetworks and embeddings options
     - Preprocessing images: cropping, mirroring, autotagging using BLIP or deepdanbooru (for anime)
- Clip skip
- Hypernetworks
- Loras (same as Hypernetworks but more pretty)
- A sparate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt. 
- Can select to load a different VAE from settings screen
- Estimated completion time in progress bar
- API
- Support for dedicated [inpainting model](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion) by RunwayML.
- via extension: [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients), a way to generate images with a specific aesthetic by using clip images embeds (implementation of [https://github.com/vicgalle/stable-diffusion-aesthetic-gradients](https://github.com/vicgalle/stable-diffusion-aesthetic-gradients))
- [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20) for instructions
- [Alt-Diffusion](https://arxiv.org/abs/2211.06679) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alt-diffusion) for instructions
- Now without any bad letters!
- Load checkpoints in safetensors format
- Eased resolution restriction: generated image's domension must be a multiple of 8 rather than 64
- Now with a license!
- Reorder elements in the UI from settings screen
- 

## Installation and Running
Make sure the required [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) are met and follow the instructions available for both [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended) and [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.

Alternatively, use online services (like Google Colab):

- [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Automatic Installation on Windows
1. Install [Python 3.10.6](https://www.python.org/downloads/windows/), checking "Add Python to PATH"
2. Install [git](https://git-scm.com/download/win).
3. Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Place stable diffusion checkpoint (`model.ckpt`) in the `models/Stable-diffusion` directory (see [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) for where to get it).
5. Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install wget git python3 python3-venv
# Red Hat-based:
sudo dnf install wget git python3
# Arch-based:
sudo pacman -S wget git python3
```
2. To install in `/home/$(whoami)/stable-diffusion-webui/`, run:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```

### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing
Here's how to add code to this repo: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation
The documentation was moved from this README over to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits
Licenses for borrowed code can be found in `Settings -> Licenses` screen, and also in `html/licenses.html` file.

- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instruct pix2pix - Tim Brooks (star), Aleksander Holynski (star), Alexei A. Efros (no star) - https://github.com/timothybrooks/instruct-pix2pix
- Security advice - RyotaK
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
