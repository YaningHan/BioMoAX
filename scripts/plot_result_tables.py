"""Render exported numerical summaries without original image data.

These are compact numerical checks, not regeneration of the composed paper figures.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("runs/table_plots"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/"results/paper/identity_curve_summary.csv")
    df=df[(df.k==5)&(df.cell=="A549")].copy()
    fig,ax=plt.subplots(figsize=(8,3.5),layout="constrained")
    ax.bar(range(len(df)),df['mean'],width=.55,color="#E15967",yerr=df.sd,
           error_kw={"ecolor":"#333333","capsize":3,"elinewidth":1})
    ax.set_xticks(range(len(df)),df.method.tolist(),fontsize=8)
    ax.set_ylabel("Top-5 hit rate (%)");ax.set_title("A549 identity retrieval (reported SD)")
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/"identity_summary.pdf");plt.close(fig)
    df=pd.read_csv(root/"results/paper/Fig5_print_curve_audit.csv")
    fig,ax=plt.subplots(figsize=(5,3.5),layout="constrained")
    for method,group in df.groupby('method',sort=False):
        group=group.sort_values('k')
        ax.plot(group.k,group['mean'],label=method)
        ax.fill_between(group.k,group.low,group.high,alpha=.15)
    ax.set(xlabel="Top K",ylabel="Queries with a hit (%)",title="A549 CRISPR-to-drug retrieval")
    ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/"crispr_curves.pdf");plt.close(fig)
    print(a.output)


if __name__=='__main__':main()
