import re
import pandas as pd
from matplotlib import font_manager
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

font_path = './Times-New-Roman.ttf'
font_manager.fontManager.addfont(font_path)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','no-latex'])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "Arial", "sans-serif"]

from scipy.stats import kendalltau, spearmanr, pearsonr
import argparse
import numpy as np

def set_ax(ax):
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

def extract_data(input_file):
    data = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
    pattern_index = re.compile(r'\[.*?\] pruned_index_pair {(.*?)}')
    pattern_pruned_loss = re.compile(r'\[.*?\] pruned_train_loss (.*?)\s')
    pattern_train_test_loss = re.compile(r'\[.*?\] final_train_loss (.*?)\sfinal_test_loss (.*?)\sfinal_test_acc (.*?)\s')

    for i in range(len(lines)):
        if pattern_index.match(lines[i]):
            index_pair = pattern_index.search(lines[i]).group(1)
            pruned_train_loss = float(pattern_pruned_loss.search(lines[i+1]).group(1))
            train_test_loss_match = pattern_train_test_loss.search(lines[i+2])
            if train_test_loss_match:
                final_train_loss = float(train_test_loss_match.group(1))
                final_test_loss = float(train_test_loss_match.group(2))
                final_test_acc = float(train_test_loss_match.group(3))/100.0
                data.append((index_pair, pruned_train_loss, final_train_loss, final_test_loss, final_test_acc))

    df_raw = pd.DataFrame(data, columns=['Index Pair', 'Pruned Train Loss', 'Final Train Loss', 'Final Test Loss', 'Final Test Acc'])

    df = df_raw.groupby('Index Pair').mean().reset_index()
    return df

def extract_l1_data(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    pattern_pruned_loss = re.compile(r'\[.*?\]: pruned_train_loss ([\d\.]+)')
    pattern_train_test_loss = re.compile(r'\[.*?\]: final_train_loss ([\d\.]+) final_test_loss ([\d\.]+) final_test_acc ([\d\.]+)')

    pruned_train_loss = []
    final_train_loss = []
    final_test_loss = []
    final_test_acc = []

    for line in lines:
        match_pruned_loss = pattern_pruned_loss.search(line)
        if match_pruned_loss:
            pruned_train_loss.append(float(match_pruned_loss.group(1)))

        match_train_test_loss = pattern_train_test_loss.search(line)
        if match_train_test_loss:
            final_train_loss.append(float(match_train_test_loss.group(1)))
            final_test_loss.append(float(match_train_test_loss.group(2)))
            final_test_acc.append(float(match_train_test_loss.group(3)))

    pruned_train_loss_mean = np.mean(pruned_train_loss) if pruned_train_loss else None
    final_train_loss_mean = np.mean(final_train_loss) if final_train_loss else None
    final_test_loss_mean = np.mean(final_test_loss) if final_test_loss else None
    final_test_acc_mean = np.mean(final_test_acc) if final_test_acc else None

    return pruned_train_loss_mean, final_train_loss_mean, final_test_loss_mean, final_test_acc_mean



def save_dataframe_to_csv(df, output_file):
    df.to_csv(output_file, index=False)


def plot_scatter(x, x_name, y, y_name, y_min, y_max, output_image_file, kendall, p_value, x_l1, y_l1, filename, x_min, x_max):
    
    fig, ax = plt.subplots(figsize=(8, 6))  # Combine the two lines for creating figure and axis
    set_ax(ax)

    # Find the index of the point with the lowest x-value
    min_x_index = x.idxmin()

    # Find the y-value corresponding to the lowest x-value
    y1 = y[min_x_index]
    tolerance = 0.00 # Tolerance for the y-value
    tolerance_for_acc = 0.000

    above_y1 = 0
    below_y1 = 0

    # Determine colors based on y1 value and count points above and below y1
    colors = []
    for value in y:
        if y_name == "Final test acc":
            if value > y1 - tolerance_for_acc:
                colors.append('red')
                above_y1 += 1
            else:
                colors.append('green')
                below_y1 += 1
        else:
            if value < y1 * (1 + tolerance):
                colors.append('red')
                below_y1 += 1
            else:
                colors.append('green')
                above_y1 += 1

    # Calculate the percentage of red points
    if y_name == "Final test acc":
        red_point_percent = above_y1 / len(y)
    else:
        red_point_percent = below_y1 / len(y)

    # Plot the scatter plot of x and y
    ax.scatter(x, y, alpha=0.6, s=35, c=colors, edgecolors=None, linewidths=0)

    # Draw the blue star point
    ax.plot(x[min_x_index], y1, '*', markersize=25, color='blue')

    # draw the l1 pruning plot 
    # ax.plot(x_l1, y_l1, 'o', markersize=10, color='purple')

    # Draw a horizontal line at y = y1
    ax.axhline(y=y1, color='blue', linestyle='--')
    ax.axvline(x=x[min_x_index], color='blue', linestyle='--')

    # Determine colors for the Kendall and p-value text
    color_for_kendall = 'black'
    color_for_pval = 'black' if p_value <= 0.05 else 'red'

    
    if y_name == "Final test acc":
        if round(kendall, 2) >= -0.2:
            color_for_kendall = 'red'
    else:
        if round(kendall, 2) <= 0.2:
            color_for_kendall = 'red'

    # Display the red point percentage
    red_text = f'Anomaly ratio: {red_point_percent*100:.2f}%'

    color_for_red = 'black' if red_point_percent < 0.5 else 'red'

    red = TextArea(red_text, textprops=dict(color=color_for_red, fontsize=26))
    combined_text = HPacker(children=[red], align="center", pad=0, sep=5)

    # Position the combined text on the plot
    anchored_text = AnchoredOffsetbox(loc='upper left', child=combined_text,
                                    pad=0., frameon=False,
                                    bbox_to_anchor=(0.35, 0.2),
                                    bbox_transform=ax.transAxes)
    ax.add_artist(anchored_text)

    # Create text areas for different colored parts
    part1 = TextArea(f'Kendall: {kendall:.2f} ', textprops=dict(color=color_for_kendall, fontsize=26))
    part2 = TextArea(f'(p-value: {p_value*100:.3f}%)', textprops=dict(color=color_for_pval, fontsize=26))

    # Pack the text areas together
    combined_text = HPacker(children=[part1, part2], align="center", pad=0, sep=5)

    # Position the combined text on the plot
    anchored_text = AnchoredOffsetbox(loc='upper left', child=combined_text,
                                    pad=0., frameon=False,
                                    bbox_to_anchor=(0.35, 0.1),
                                    bbox_transform=ax.transAxes)
    ax.add_artist(anchored_text)


    # Set labels and grid
    if y_name == "Final test acc":
        y_name = "Final test accuracy"
    ax.set_xlabel(x_name, fontsize=26)
    ax.set_ylabel(y_name, fontsize=26)


    # ax.set_xlim(x_min, x.max()+0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max) 
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(output_image_file)
    plt.close()

def calculate_kendall_coefficient(x, y):
    tau, p_value = kendalltau(x, y)
    return tau, p_value

def calculate_spearmanr(x, y):
    rho, p_value = spearmanr(x, y)
    return rho, p_value

def calculate_pearsonr(x, y):
    r, p_value = pearsonr(x, y)
    return r, p_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a summary text file and generate scatter plots.')
    parser.add_argument('--filename', type=str, help='The base name of the summary input file (without path and extension)')
    parser.add_argument('--y_min', type=float, help='The minimum value of the y-axis')
    parser.add_argument('--y_max', type=float, help='The maximum value of the y-axis for the second scatter plot')
    parser.add_argument('--test_acc_y_min', type=float, help='The minimum value of the y-axis for the test accuracy scatter plot')
    parser.add_argument('--test_acc_y_max', type=float, help='The maximum value of the y-axis for the test accuracy scatter plot')
    parser.add_argument('--x_min', type=float, help='The minimum value of the x-axis')
    parser.add_argument('--x_max', type=float, help='The maximum value of the x-axis')
    args = parser.parse_args()

    import os
    if not os.path.exists("./Pruning/record/figures"):
        os.makedirs("./Pruning/record/figures")

    input_file_path = f'./Pruning/record/summary/{args.filename}_summary.txt'
    input_l1_file_path = f'./Pruning/record/summary/{args.filename}-l1-summary.txt'
    output_image_path2 = f'./Pruning/record/figures/{args.filename}_scatterplot2.pdf'
    output_image_path3 = f'./Pruning/record/figures/{args.filename}_scatterplot3.pdf'

    df = extract_data(input_file_path)
    pruned_train_loss, final_train_loss, final_test_loss, final_test_acc = extract_l1_data(input_l1_file_path)



    print(f"\nAnalysis for {args.filename}")
    print("     Pruned Train Loss vs Final Test Loss & Final Test Acc: ")

    tau, p_value = calculate_kendall_coefficient(df['Pruned Train Loss'], df['Final Test Loss'])
    plot_scatter(df['Pruned Train Loss'], "Pruned train loss", df['Final Test Loss'], "Final test loss", args.y_min, args.y_max, output_image_path2, tau, p_value, pruned_train_loss, final_test_loss, args.filename, args.x_min, args.x_max)
    print(f"{tau:.2f} / {p_value:.1e}")


    tau, p_value = calculate_kendall_coefficient(df['Pruned Train Loss'], df['Final Test Acc'])
    plot_scatter(df['Pruned Train Loss'], "Pruned train loss", df['Final Test Acc'], "Final test acc", args.test_acc_y_min, args.test_acc_y_max, output_image_path3, tau, p_value, pruned_train_loss, final_test_acc/100, args.filename, args.x_min, args.x_max)
    print(f"{tau:.2f} / {p_value:.1e}")