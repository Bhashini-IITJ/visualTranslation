#!/bin/bash
m2m=false
hin_eng=false
de=false

while [ "$1" != "" ]; do
    case $1 in
    "-i")
        shift
        input_folder=$1
        ;;
    "-o")
        shift
        output_folder=$1
        ;;
    "-f")
        shift
        input_file=$1
        ;;
    "--eng_kor")
        eng_kor=true #한영
        ;;
    "--de")
        de=true
        ;;
    esac
    shift
done

mkdir -p tmp

## paragraph detection
conda activate itv2_hf

if [ "$de" = true ]; then
    python exclude_key_words.py --file "$input_file" 
    python detect_para.py
else
    cp "$input_file" tmp/i_s_info.json
    cp "$input_file" tmp/para_info.json
    python form_para_info.py
fi

## translation
if [ "$de" = true ]; then
    if [ "$kor_eng" = true ]; then
        python translate_de.py 
    else
        python translate_de.py --eng_to_kor
    fi
    python form_word_crops.py
else
    if [ "$kor_eng" = true ]; then
        python translate.py 
    else
        python translate.py --eng_to_kor
    fi
fi

## cropping i_s
conda deactivate
conda activate srnet_plus_2
python generate_crops.py --folder "$input_folder"

## modifying i_s
python modify_crops.py

## creating i_t
python generate_i_t.py

## scene text eraser
conda deactivate














# conda activate scene_text_eraser
# python make_masks.py --folder "$input_folder"
# python scene_text_eraser.py --folder "$input_folder"

# ## generating modified images
# python make_output_base.py --folder "$input_folder"

# ## generating bg
# python make_bg.py

# ## infer srnet_plus_2
# conda deactivate
# conda activate srnet_plus_2
# python generate_o_t.py

# ## blend the crops
# python blend_o_t_bg.py

# ## generate the final output
# python create_final_images.py --output_folder "$output_folder"
# # rm -r tmp
