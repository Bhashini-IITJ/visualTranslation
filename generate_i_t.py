import json
import os
from tqdm import tqdm

labels = json.load(open('./tmp/para_info.json'))

save_path = './tmp/i_t'
temp_save_path = './tmp/i_t_temp'
os.makedirs(save_path, exist_ok=True)
os.makedirs(temp_save_path, exist_ok=True)

text_image_size = '256x100'
image_size = '256x128'

gray_bg_path = './i_t_utils/gray_bg_256x128.png'
font_name = 'Noto Sans'

for crop_name, label in tqdm(labels.items()):
    
    label = label['trans_txt']
    
    save_file_path = os.path.join(save_path, f'{crop_name}.png')
    temp_save_file_path = os.path.join(temp_save_path, f'{crop_name}.png')
    
    input_text_command = f'convert -alpha set -background "rgb(121,127,141)" pango:\'  \
                        <span font_stretch="semicondensed" foreground="#000000" font=" {font_name} 30 ">{label.lower()}</span> \
                        \' \png:-|convert -  \\( +clone \\) +swap  -background "rgb(121,127,141)" -layers merge  +repage png:-|\
                        convert -   -trim +repage -resize {text_image_size} {temp_save_file_path}'

    os.system(input_text_command.encode('utf-8'))
                        
    finalInputTextCommand = 'composite -gravity Center ' + f'{temp_save_file_path}' + ' ' + gray_bg_path
    finalInputTextCommand += ' png:-|'
    finalInputTextCommand += 'convert -  ' + save_file_path
    os.system(finalInputTextCommand.encode('utf-8'))
os.system(f'rm -rf {temp_save_path}')