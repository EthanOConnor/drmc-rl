"""Quick plotting utility for seed_sweep parquet/CSV outputs."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', type=str, default='eval/seed_metrics.parquet')
    args = ap.parse_args()
    df = pd.read_parquet(args.infile)
    df = df.sort_values('E_T')
    plt.figure()
    plt.plot(df['seed'], df['E_T'], marker='o')
    plt.title('Mean frames to clear per seed')
    plt.xlabel('seed')
    plt.ylabel('E[T] (frames)')
    plt.tight_layout()
    out = args.infile.replace('.parquet', '_mean_frames.png')
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

