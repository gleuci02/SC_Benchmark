import matplotlib.pyplot as plt

def plot_metrics(algorithm_names, acc_scores, nmi_scores, output_file="results/metrics_plot.png"):
    """
    Plot ACC and NMI scores for each algorithm.

    Parameters:
    - algorithm_names: list of str, Names of the algorithms.
    - acc_scores: list of float, ACC scores for each algorithm.
    - nmi_scores: list of float, NMI scores for each algorithm.
    - output_file: str, Path to save the plot (default: "results/metrics_plot.png").
    """
    x = range(len(algorithm_names))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    width = 0.4  # Bar width

    # Plot ACC and NMI side by side
    plt.bar(x, acc_scores, width=width, label="ACC", color="skyblue", edgecolor="black")
    plt.bar([i + width for i in x], nmi_scores, width=width, label="NMI", color="orange", edgecolor="black")

    # Add labels, title, and legend
    plt.xlabel("Algorithms", fontsize=12)
    plt.ylabel("Scores", fontsize=12)
    plt.title("Clustering Metrics: ACC and NMI", fontsize=14)
    plt.xticks([i + width / 2 for i in x], algorithm_names, fontsize=10)
    plt.ylim(0, 1.1)  # Scores are typically between 0 and 1
    plt.legend(fontsize=12)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()