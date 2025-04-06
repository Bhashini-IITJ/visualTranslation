#!/bin/bash
# Setup the entire system for visual translation with GPU optimization

## Ensure conda is properly initialized
source $(conda info --base)/etc/profile.d/conda.sh

## ✅ Translation Environment Setup
root_dir=$(pwd)
conda create -n itv2_hf python=3.9 -y
conda activate itv2_hf

# ✅ Ensure CUDA environment is correctly set for GPU
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# ✅ Install essential packages
conda install pip -y
python -m pip install --upgrade pip
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# ✅ Install other dependencies
python -m pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
python -c "import nltk; nltk.download('punkt')"

# ✅ Install prebuilt flash-attn (NO CPU BUILD!)
pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

python -m pip install bitsandbytes scipy accelerate datasets --no-cache-dir
python -m pip install sentencepiece

# ✅ Clone and install IndicTransToolkit
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
python -m pip install --editable ./
pip install scipy
cd $root_dir
conda deactivate

# ## ✅ Scene Text Eraser Environment Setup
# git clone https://github.com/Onkarsus13/Diff_SceneTextEraser.git
# conda create -n scene_text_eraser python=3.9 -y
# conda activate scene_text_eraser

# cd Diff_SceneTextEraser
# pip install -e ".[torch]"
# pip install -e .[all,dev,notebooks]
# pip install jax==0.4.23 jaxlib==0.4.23
# pip install "huggingface_hub<0.26.0"
# cd $root_dir
# conda deactivate

## ✅ SRNet Environment Setup
conda create -n SRNet python=3.9.20 -y
conda activate SRNet
pip install -r SRNet/requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

conda deactivate

## ✅ Install Image Processing Libraries
sudo apt update
sudo apt install -y libpango1.0-dev libcairo2-dev imagemagick

## ✅ Final Check: Ensure GPU is recognized
conda activate itv2_hf
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
conda deactivate








echo "🎉 Setup complete! Everything is ready to use with GPU acceleration."
