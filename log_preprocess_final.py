import argparse
import os
import glob
import re
from collections import defaultdict
from scipy.stats import kendalltau
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import font_manager
font_path = './Times-New-Roman.ttf'
font_manager.fontManager.addfont(font_path)
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','no-latex'])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "Arial", "sans-serif"]

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
            int(re.search('batch(\d+)\.txt', x).group(1))  # 匹配 batch 后的数字
        ))
    else:
        files = glob.glob(os.path.join(directory, f"{prefix}_t*_batch*.txt"))
        files.sort(key=lambda x: (
            int(re.search('_t(\d+)_', x).group(1)),  # 匹配 t 后的数字
            int(re.search('batch(\d+)\.txt', x).group(1))  # 匹配 batch 后的数字
        ))

    return files

def extract_and_split_lines(content1, content2, prefix):

    pattern_final = re.compile(r'^\[\d+ \d+ \d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\] \[(\d+)/(\d+)\] final_train_loss .* final_test_acc (\d+\.\d+)')
    # get final train loss
    pattern_final2 = re.compile(r'^\[\d+ \d+ \d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\] \[(\d+)/(\d+)\] final_train_loss (\d+\.\d+)')

    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    
    acc_results = []


    current_index1 = None
    current_index2 = None
    acc1_value1 = []

    final_test_acc1 = None
    final_test_acc2 = None
    acc1_value2 = []

    for line in lines1:
        match_final1 = pattern_final.match(line)

        if match_final1:
            current_index1 = int(match_final1.group(1))
            final_test_acc1 = float(match_final1.group(3))
            acc1_value1.append((current_index1, final_test_acc1))

    for line in lines2:
        match_final2 = pattern_final.match(line)

        if match_final2:
            current_index2 = int(match_final2.group(1))
            final_test_acc2 = float(match_final2.group(3))
            acc1_value2.append((current_index2, final_test_acc2))

    # combine the results into one list(index, final_test_acc1, final_test_acc2)
    acc_results = []
    for i in range(len(acc1_value1)):
        acc_results.append((acc1_value1[i][0], acc1_value1[i][1], acc1_value2[i][1]))

    # Group by index and calculate the average
    grouped_results = defaultdict(list)
    for index, final_test_acc1, final_test_acc2 in acc_results:
        grouped_results[index].append((final_test_acc1, final_test_acc2))


    # calculate the average of final_test_acc1 and final_test_acc2 for each index
    average_results = []
    for item in grouped_results.items():
        index = item[0]
        data = item[1]
        avg_final_test_acc1 = sum([x[0] for x in data]) / len(data)
        avg_final_test_acc2 = sum([x[1] for x in data]) / len(data)

        average_results.append((index, avg_final_test_acc1, avg_final_test_acc2))
        

    # calculate the Kendall's tau
    final_test_acc1 = [x[1] for x in average_results]
    final_test_acc2 = [x[2] for x in average_results]

    tau, p_value = kendalltau(final_test_acc1, final_test_acc2)
    # draw the plot for final_test_acc1 and final_test_acc2
    draw_plot(final_test_acc1, final_test_acc2, prefix, tau, p_value)

    print(f"{prefix}, 10%epoch kendall: {tau:.2f}/{p_value:.1e}")

def set_ax(ax):
    '''This will modify ax in place.
    '''
    # set background
    # ax.grid(color='white')
    # ax.set_facecolor('whitesmoke')

    # remove axis line
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # remove tick but keep the values
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.xaxis.set_ticks_position('none')
    # ax.yaxis.set_ticks_position('none')

def draw_plot(final_test_acc1, final_test_acc2, prefix, tau, p_value):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    set_ax(ax)

    text = f"Kendall: {tau:.2f} (p-value: {p_value:.1e})"
    red = TextArea(text, textprops=dict(color='black', fontsize=26))
    combined_text = HPacker(children=[red], align="center", pad=0, sep=5)

    # Position the combined text on the plot
    anchored_text = AnchoredOffsetbox(loc='upper left', child=combined_text,
                                    pad=0., frameon=False,
                                    bbox_to_anchor=(0.35, 0.1),
                                    bbox_transform=ax.transAxes)
    ax.add_artist(anchored_text)

    ax.scatter(final_test_acc2, final_test_acc1, alpha=0.6, s=35, edgecolors=None, linewidths=0, color = 'blue')


    ax.set_ylabel("Test accuracy (100% epochs)", fontsize=26)
    ax.set_xlabel("Test accuracy (10% epochs)", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f'./Pruning/record/figures/{prefix}_acc.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Merge text files based on a given prefix and generate a summary.")
    parser.add_argument("--filename", type=str, required=True, help="Prefix of the files to merge")
    args = parser.parse_args()

    directory = "./Pruning/record"

    if not os.path.exists("./Pruning/record/figures"):
        os.makedirs("./Pruning/record/figures")

    input_files1 = find_files(args.filename, directory)
    merged_content1 = merge_files(input_files1)

    
    another_filename = "3epoch_" + args.filename

    if "vgg" in args.filename or "resnet56" in args.filename:
        another_filename = "12epoch_" + args.filename

    input_files2 = find_files(another_filename, directory)

    # if no files found for another_filename, just return
    if not input_files2:
        print(f"No files found for {another_filename}")
        return
    
    merged_content2 = merge_files(input_files2)

    extract_and_split_lines(merged_content1, merged_content2, args.filename)

if __name__ == "__main__":
    main()