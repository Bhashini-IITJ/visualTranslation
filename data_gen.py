import os
import argparse
import concurrent.futures

def run_command_eng_hin(i, n):
    os.system(f"python render_Indian_language_scenetext.py ./dataset\ utils/eng_hin.txt ./dataset\ utils/Hin_Uniq_RS.txt ./dataset/o{i} {n}")

def run_command_hin_eng(i, n):
    os.system(f"python render_Indian_language_scenetext.py ./dataset\ utils/hin_eng.txt ./dataset\ utils/Hin_Uniq_RS.txt ./dataset/o{i} {n}")

parser = argparse.ArgumentParser()
parser.add_argument("--per_loop", type=int, help="Number of iterations")
parser.add_argument("--num_loops", type=int, help="Number of loops")
parser.add_argument("--hin_eng", type=str, help="Path to hin_eng.txt")
args = parser.parse_args()
n = args.per_loop
n_loops = args.num_loops

if args.hin_eng == "true":
    run_command = run_command_hin_eng
else:
    run_command = run_command_eng_hin

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_command, i, n) for i in range(n_loops)]
    
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
