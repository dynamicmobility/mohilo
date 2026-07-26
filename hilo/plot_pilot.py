import argparse
import matplotlib.pyplot as plt
import pandas as pd

def main(path_to_csv):
    # Load the CSV file
    df = pd.read_csv(path_to_csv)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(list(range(len(df['Cost']))), df['Cost'], c='blue', alpha=0.5)
    plt.title('Scatter Plot of Data')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.grid(True)
    save_path = 'hilo/output/pilot-mohilo-raw.pdf'
    plt.savefig('hilo/output/pilot-mohilo-raw.pdf')  # Save the plot as a PNG file
    print('Saved to', save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from a CSV file')
    parser.add_argument('--csv', default='human_data/pilot_mohilo.csv', help='Path to the CSV file')
    args = parser.parse_args()
    main(args.csv)