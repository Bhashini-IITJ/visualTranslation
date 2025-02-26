import re
import json
import argparse

perser = argparse.ArgumentParser()
perser.add_argument("--file", type=str, required=True)
args = perser.parse_args()
file = args.file


tlds = (
    "com", "org", "net", "int", "edu", "gov", "mil", "info", "biz", "name", "museum", "coop", "aero", "xxx", "idv"
)

def exclude(s):
    if re.fullmatch(r'\d+(\.\d+)?', s):
        return True
    if s.startswith("www.") or re.fullmatch(r'(https?://)?(?:[-\w.]|(?:%[\da-fA-F]{2}))+(\.(?:' + '|'.join(tlds) + '))', s):
        return True
    return bool(re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s))

data = json.load(open(file,'r'))
final_data = {k: v for k, v in data.items() if not exclude(v["txt"])}
json.dump(final_data, open("tmp/i_s_info.json",'w'), indent=4)

print(f"<<<<<파일 확인>>>>> exclude_key_words.py")