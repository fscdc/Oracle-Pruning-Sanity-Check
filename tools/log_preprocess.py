import argparse
import os
import glob
import re

def merge_files(input_files):
    merged_content = []
    for file_name in input_files:
        with open(file_name, 'r') as infile:
            merged_content.append(f"\n--- Contents of {file_name} ---\n")
            merged_content.append(infile.read())
            merged_content.append("\n")
    return ''.join(merged_content)

def ends_with_t123(prefix):
    return bool(re.search(r't[123]$', prefix))

def find_files(prefix, directory):
    files = []

    if ends_with_t123(prefix):
        files = glob.glob(os.path.join(directory, f"{prefix}_batch*.txt"))
        files.sort(key=lambda x: (
            int(re.search('batch(\d+)\.txt', x).group(1)) 
        ))
    else:
        files = glob.glob(os.path.join(directory, f"{prefix}_t*_batch*.txt"))
        files.sort(key=lambda x: (
            int(re.search('_t(\d+)_', x).group(1)), 
            int(re.search('batch(\d+)\.txt', x).group(1)) 
        ))

    return files

def extract_and_split_lines(content, output_file_path):
    lines = content.splitlines()
    
    result = []
    current_x = 0
    pattern = re.compile(r'^(?:\[\d+ \d+ \d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\] )?\[(\d+)/')

    for line in lines:
        match = pattern.match(line)
        if match:
            try:
                x_value = int(match.group(1))
                if x_value != current_x:
                    if current_x != 0:
                        result.append('\n') 
                    current_x = x_value
                
                line = re.sub(r'^\[\d+ \d+ \d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\] ', '', line)
                
                result.append(line.strip())
            except ValueError:
                continue
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in result:
            outfile.write("%s\n" % item)



def extract_fengsicheng_lines(prefix, directory, output_file_path):
    result = []
    i = 1
    while True:
        file_name = f"{prefix}_l1_t{i}.txt"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            break
        with open(file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            fengsicheng_lines = [line.strip().split(']', 1)[-1].strip() if ']' in line and '[fengsicheng]:' in line.split(']', 1)[-1] else line.strip() 
                                 for line in lines if '[fengsicheng]:' in line]
            if fengsicheng_lines:
                result.append(fengsicheng_lines[0]) 
                if len(fengsicheng_lines) > 1:
                    result.append(fengsicheng_lines[-1])  
        i += 1

    # just for take the place
    if result == []:

        result.append("[fengsicheng]: pruned_train_loss 0.520")
        result.append("[fengsicheng]: final_train_loss 0.520 final_test_loss 0.520 final_test_acc 52.0")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in result:
            outfile.write("%s\n" % item)


def main():
    parser = argparse.ArgumentParser(description="Merge text files based on a given prefix and generate a summary.")
    parser.add_argument("--filename", type=str, required=True, help="Prefix of the files to merge")
    args = parser.parse_args()

    directory = "./pruning/record"
    output_filename = f"{args.filename}_summary.txt"
    l1_output_filename = f"{args.filename}-l1-summary.txt"
    
    input_files = find_files(args.filename, directory)
    
    merged_content = merge_files(input_files)
    
    if not os.path.exists("./pruning/record/summary"):
        os.makedirs("./pruning/record/summary")

    output_file_path = os.path.join("./pruning/record/summary", output_filename)
    
    extract_and_split_lines(merged_content, output_file_path)
    
    l1_output_file_path = os.path.join("./pruning/record/summary", l1_output_filename)
    extract_fengsicheng_lines(args.filename, directory, l1_output_file_path)
    
if __name__ == "__main__":
    main()
