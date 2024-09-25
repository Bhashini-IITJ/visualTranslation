<h1 align='center'>Show Me the World in My Language: Establishing the First Baseline for Scene-Text to Scene-Text Translation (Official code)</h1>
<p align='center'>
    <a href="https://icpr2024.org/"><img src="https://img.shields.io/badge/ICPR-2024-4b44ce"></a>
    <a href="https://arxiv.org/abs/2308.03024"><img src="https://img.shields.io/badge/Paper-pdf-red"></a>
    <a href="https://github.com/Bhashini-IITJ/visualTranslation/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
    <a href="https://vl2g.github.io/projects/visTrans/"><img src="https://img.shields.io/badge/Project-page-green"></a>
</p>
Implementation of Baseline for Scene Text-to-Scene Text Translation

<img src="assets/welcome.png" width="100%">

# Inference on datasets used 
This release only supports inference on datasets used in the paper, i.e., BSTD and ICDAR 2013, and using precomputed scene text detection and recognition. Please follow the below instructions for inference on our VT-Real dataset. For detailed information for specific tasks check the [training](#training) section 

1. Clone the repo and set up the required dependencies
    ```bash
    git clone https://github.com/Bhashini-IITJ/visualTranslation.git
    source ./setup.sh
    ```

2. Download the input VT-Real images (which are to be translated) (download details in the [Project page](https://vl2g.github.io/projects/visTrans/)) and put them in folders **source_eng** (ICDAR images) and **source_hin**  (BSTD images) in the project directory.

3. Download the translation checkpoints [eng_hin.model](https://drive.google.com/file/d/1OqloAgsdf-L9hmoeYCW3qrLdtNTQJisx/view?usp=sharing) and [hin_eng.model](https://drive.google.com/file/d/1qb9aUjgGp53lJdfLPUnCVb7mEbd5-gNi/view?usp=sharing) and put them in a folder named **model** inside the project directory.

4. We need an "i_s_info.json" file carrying the information of word-level bounding boxes.Different files are required for different languages.
5. Download the json file from the below table based on the language, rename it as i_s_info.json and place it in the project directory


| **Language** | **B-7** |
| :---: | :---: |

| Eng -> Hin | [DBNet+Parseq](https://drive.google.com/file/d/1S8ayCLhO2EugF3CLQnHm9J7jJEAq8Hr_/view?usp=drive_link) |

| Hin -> Eng | [Oracle](https://drive.google.com/file/d/1F_IddWKhw4C4UXOEzH-8a3_4VNqCTias/view?usp=sharing) |

6. Then run one of the below commands based on the required baselines and language translation direction
  ### Eng - Hin
  #### B7
  ```bash
  source ./infer.sh -i source_eng -o output -f i_s_info.json --de
  ```
  ### Hin - Eng
  #### B7
  ```bash
  source ./infer.sh -i source_hin  -o output -f i_s_info.json --de --hin_eng
  ```


In both cases a new folder named **output** will be created and the translated images will be saved in it.
  
# Training 
## Dataset generation
The dataset generation script is designed for ImageMagick v6 but can also work with ImageMagick v7, although you may encounter several warnings. The dataset can be generated for either English-to-Hindi (eng-hin) or Hindi-to-English (hin-eng) translations.
### Setup Instructions:
1. Download [this](https://drive.google.com/drive/folders/1Kf4RhqNQ6SP_YJALgWUMG0gvAkbK8S25) folder and add it to this repository.
2. Unzip all the files within the folder.
3. Install the fonts located in the devanagari.zip file.
   
### Generating the Dataset:
To generate the dataset, run the following command:
```bash
./dataset_gen.sh [ --num_workers <number of loops> --per_worker <number of samples per loop> --hin_eng]
```
Command Options:
--num_workers: Specifies the number of workers for dataset generation. Default: 20.
--per_worker: Specifies the number of samples per loop. Default: 3000.
--hin_eng: Generates a Hindi-to-English (hin-eng) dataset. If not specified, the dataset will be generated for English-to-Hindi (eng-hin).
Note: To generate a dataset for other language pairs, modify the commands in data_gen.py accordingly.

## Training SRNet++

SRNet++ can be trained with the following command:
```bash
conda activate srnet_plus_2
python train_o_t.py
```
change the path of 'data_dir' parameter in cfg.py file if you are using dataset with different path than default.

SRNet++ can be infered with following command lines:
```bash
conda activate srnet_plus_2
python generate_o_t.py
```
please change the path according to your use case. The inputs for the inferece are i_s and i_t. Example given below.
|**i_s**|**i_t**|
|:--:|:--:|
|![](assets/i_s.png)|![](assets/i_t.png)|

## 2. scene-text detection/recognition
In this project use SOTA scene text detection/recognition model, To form the json file which contains the information of what and where is the word. The Scene Text Detection and Recognition models are DBNet and ParSeq respectively. The file has to be in the following format:

```json
{
    "100_0":{
        "txt": "hello",
        "bbox": [x1, y1, x2, y2]
    },
    ...
}
```


## Warning and troubleshooting
- please make sure that imagemagick support png format after the setup.
- Data generation code is written for imagemagickv6. It would work for imagemagickv7 but you will have a lots of warnings. 
## Bibtex (how to cite us)
```
@InProceedings{vistransICPR2024,
    author    = {Vaidya, Shreyas and Sharma, Arvind Kumar and Gatti, Prajwal and Mishra, Anand},
    title     = {Show Me the World in My Language: Establishing the First Baseline for Scene-Text to Scene-Text Translation},
    booktitle = {ICPR},
    year      = {2024},
}
```

## Acknowledgements
1. [SRNet](https://github.com/lksshw/SRNet)
2. [indic scene text render](https://github.com/mineshmathew/IndicSceneTextRendering)
3. [Scene text eraser](https://github.com/Onkarsus13/Diff_SceneTextEraser)
4. [facebook-m2m](https://huggingface.co/facebook/m2m100_418M)
5. [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
6. [Contrique]()

## Contact info
1. Arvind Kumar Sharma - arvindji0201@gmail.com
