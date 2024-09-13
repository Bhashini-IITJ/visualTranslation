#!/bin/bash
root_dir=$(pwd)
conda create -n itv2_hf python=3.9 -y
conda activate itv2_hf
conda install pip
python -m pip install --upgrade pip
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python -c "import nltk; nltk.download('punkt')"
python -m pip install bitsandbytes scipy accelerate datasets flash-attn>=2.1
python -m pip install sentencepiece
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
python -m pip install --editable ./
cd $root_dir
