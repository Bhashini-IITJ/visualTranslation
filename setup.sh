#/bin/bash
# setup the entire system for the visual translation

## translation environment
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface
./install.sh
cd ../..

## scene text erasor environment
conda env create -f scene_text_eraser.yml

## vtnet environment
conda env create -f vtnet.yml


## imagemagick, pango,cairo,pangocairo

sudo apt update
sudo apt install libpango1.0-dev libcairo2-dev imagemagick


