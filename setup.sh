#/bin/bash
# setup the entire system for the visual translation

## translation environment
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
pip install scipy
cd $root_dir
conda deactivate

## scene text erasor environment
git clone https://github.com/Onkarsus13/Diff_SceneTextEraser.git
conda create -n scenee_text_erasor python=3.9 -y
conda activate scene_text_erasor
cd Diff_SceneTextEraser
pip install -e ".[torch]"
pip install -e .[all,dev,notebooks]
pip install jax==0.4.23 jaxlib==0.4.23
cd $root_dir
conda deactivate

## srnet_plus_2 environment
conda create -n srnet_plus_2 python=3.8.0 -y
conda activate srnet_plus_2
pip install -r srnet_plus_2.txt
conda deactivate


## imagemagick, pango,cairo,pangocairo

sudo apt update
sudo apt install libpango1.0-dev libcairo2-dev imagemagick


