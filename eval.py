import matplotlib.pyplot as plt
from model_loader import load_models
from data.dataset import pairs
from training.metrics import *
from collections import defaultdict

def plot_with_values(labels, values, title, filename):
    plt.bar(labels, values)
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title(title)

    # Add values on top of each bar
    for label, value in zip(labels, values):
        plt.text(label, value, f'{value:.2f}', ha='center', va='bottom')

    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':
    toPhonMetric = defaultdict(list)
    toWordMetric = defaultdict(list)

    toPhon, toChar = load_models()
    for i, j in pairs[:1000]:
        word = toChar.translate(j)
        phon = toPhon.translate(i)
        word.pop()
        phon.pop()

        toPhonMetric['jaccard'].append(jaccard_similarity(phon, j))
        toPhonMetric['edit_distance'].append(edit_distance(phon, j))
        toPhonMetric['overlap'].append(overlap_coefficient(phon, j))
        toPhonMetric['dice'].append(dice_coefficient(phon, j))

        toWordMetric['jaccard'].append(jaccard_similarity(word, i))
        toWordMetric['edit_distance'].append(edit_distance(word, i))
        toWordMetric['overlap'].append(overlap_coefficient(word, i))
        toWordMetric['dice'].append(dice_coefficient(word, i))

    # Calculate averages and plot with values for toPhonMetric
    averages_to_phon = {key: sum(values) / len(values) for key, values in toPhonMetric.items()}
    plot_with_values(averages_to_phon.keys(), averages_to_phon.values(), 'Average Metrics - Word to Phonemes', 'eval_to_phon.png')

    # Calculate averages and plot with values for toWordMetric
    averages_to_word = {key: sum(values) / len(values) for key, values in toWordMetric.items()}
    plot_with_values(averages_to_word.keys(), averages_to_word.values(), 'Average Metrics - Phonemes to Word', 'eval_to_word.png')
