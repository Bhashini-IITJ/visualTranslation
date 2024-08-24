import argparse
import json

import torch
from IndicTransTokenizer import IndicProcessor
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

parser = argparse.ArgumentParser()
parser.add_argument("--eng_to_hin",action="store_true")
parser.add_argument("--m2m")
args = parser.parse_args()
mode = args.eng_to_hin
trans_model = args.m2m
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if trans_model != "true":
    if mode:
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    else:
        model_name = "ai4bharat/indictrans2-indic-en-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    ip = IndicProcessor(inference=True)
    model = model.to(DEVICE)

    # Set the source and target languages
    if mode:
        src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    else:
        src_lang, tgt_lang = "hin_Deva", "eng_Latn"

    img2info = json.load(open('tmp/para_info.json'))

    # Translate each para in the list
    for img_id in tqdm(img2info.keys()):
        img_info = img2info[img_id]
        word = img_info['txt']
        
        # Set the source language
        tokenizer.src_lang = src_lang

        batch = ip.preprocess_batch(
            [word],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        
        # Tokenize and encode the source text
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translation = ip.postprocess_batch(generated_tokens, lang=tgt_lang)[0]
        img2info[img_id]['trans_txt'] = translation
else:
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # Set the source and target languages
    if mode:
        src_lang = "en"
        target_lang = "hi"
    else:
        src_lang = "hi"
        target_lang = "en"

    # List of English words to translate
    img2info = json.load(open("tmp/para_info.json"))

    # Initialize an empty list to store translations
    translations = {}

    # Translate each word in the list
    for img_id in tqdm(img2info.keys()):
        img_info = img2info[img_id]
        word = img_info['txt']
        
        # Set the source language
        tokenizer.src_lang = src_lang
        
        # Tokenize and encode the source text
        encoded_src = tokenizer(word.lower().strip(), return_tensors="pt").to(DEVICE)
        
        # Generate translations
        generated_tokens = model.generate(**encoded_src, forced_bos_token_id=tokenizer.get_lang_id(target_lang)).to("cpu")
        
        # Decode and append the translation to the list
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        img2info[img_id]['trans_txt'] = translation

json.dump(img2info,open("tmp/para_info.json",'w'),indent=4)

print("Translation completed.")