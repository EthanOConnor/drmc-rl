"""Quick plotting utility for seed_sweep parquet/CSV outputs."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', type=str, default='eval/seed_metrics.parquet')
    args = ap.parse_args()
    df = pd.read_parquet(args.infile)
    df = df.sort_values('E_T')
    fig, ax = plt.subplots()
    ax.plot(df['seed'], df['E_T'], marker='o')
    ax.set(
        title='Mean frames to clear per seed',
        xlabel='seed',
        ylabel='E[T] (frames)',
    )
    fig.tight_layout()
    out = args.infile.replace('.parquet', '_mean_frames.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
