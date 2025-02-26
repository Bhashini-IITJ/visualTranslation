import argparse
import json

import torch
from IndicTransToolkit.IndicTransToolkit import IndicProcessor # 수정
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

parser = argparse.ArgumentParser()
parser.add_argument("--eng_to_kor",action="store_true") #수정
args = parser.parse_args()
mode = args.eng_to_kor # 영->한
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mode=True
if mode: #영한
    model_name = "NHNDQ/nllb-finetuned-en2ko"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    
else:#한영
    model_name = "Helsinki-NLP/opus-mt-ko-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = MarianMTModel.from_pretrained(model_name, trust_remote_code=True)

ip = IndicProcessor(inference=True)
model = model.to(DEVICE)
model.eval()
# Set the source and target languages
if mode:
    src_lang, tgt_lang = "eng_Latn", "kor_Hang"
else:
    src_lang, tgt_lang = "kor_Hang", "eng_Latn"

img2info = json.load(open('tmp/para_info.json'))
cnt=0
# Translate each para in the list
for img_id in tqdm(img2info.keys()):
    img_info = img2info[img_id]
    
    for i in range(len(img_info['para'])):
        word = img_info['para'][i]['txt']
        
        # # Set the source language
        # tokenizer.src_lang = src_lang

        # batch = ip.preprocess_batch(
        #     [word],
        #     src_lang=src_lang,
        #     tgt_lang=tgt_lang,
        # )
        
        # # Tokenize and encode the source text
        # inputs = tokenizer(
        #     batch,
        #     truncation=True,
        #     padding="longest",
        #     return_tensors="pt",
        #     return_attention_mask=True,
        # ).to(DEVICE)
        
        # # Generate translations
        # with torch.no_grad():
        #     generated_tokens = model.generate(
        #         **inputs,
        #         use_cache=True,
        #         min_length=0,
        #         max_length=256,
        #         num_beams=5,
        #         num_return_sequences=1,
        #     )

        # # Decode the generated tokens into text
        # with tokenizer.as_target_tokenizer():
        #     generated_tokens = tokenizer.batch_decode(
        #         generated_tokens.detach().cpu().tolist(),
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=True,
        #     )
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)

        translated = model.generate(**inputs.to(DEVICE))
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Postprocess the translations, including entity replacement
        # translation = ip.postprocess_batch(generated_tokens, lang=tgt_lang)[0]
        img2info[img_id]['para'][i]['trans_txt'] = translation
        if(len(translation)==0):cnt+=1
json.dump(img2info,open("tmp/para_info.json",'w'),indent=4)

# print(cnt)
print("Translation completed.")
print(f"<<<<<파일 확인>>>>> translate_de.py")
