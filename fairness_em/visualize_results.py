"""
Phase 3: Results Visualization and Analysis

This module generates publication-quality visualizations and statistical analysis
of the Phase 3 loss curve exploration results.

Key Outputs:
1. Pareto frontier plot (Accuracy vs Fairness)
2. Alpha sensitivity analysis
3. Comparison with Phase 1 baseline
4. Statistical summary and recommendations

Supports the run artifact contract: looks for results inside
results/phase3/<run_id>/ folders.

Usage:
    python visualize_results.py [--run_dir path/to/run_folder]
    python visualize_results.py [--results_file path/to/results.csv]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for safe file saves
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def plot_pareto_frontier(results_df, save_path=None, show_plot=True):
    """
    Generate the main Pareto frontier plot showing accuracy-fairness trade-off.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Main trade-off curve
    ax.plot(results_df['ppvp_disparity'], results_df['f1_score'],
           'bo-', linewidth=2.5, markersize=10, label='Accuracy-Fairness Trade-off')

    # Annotate points with alpha values
    for _, row in results_df.iterrows():
        ax.annotate(f'α={row["alpha"]:.1f}',
                   (row['ppvp_disparity'], row['f1_score']),
                   xytext=(8, -3), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    # Highlight key points
    baseline_idx = results_df['alpha'].idxmin()  # α=0.0
    pure_fair_idx = results_df['alpha'].idxmax()  # α=1.0

    ax.plot(results_df.loc[baseline_idx, 'ppvp_disparity'],
           results_df.loc[baseline_idx, 'f1_score'],
           'rs', markersize=15, label='Baseline (α=0)')

    ax.plot(results_df.loc[pure_fair_idx, 'ppvp_disparity'],
           results_df.loc[pure_fair_idx, 'f1_score'],
           'g^', markersize=15, label='Pure Fairness (α=1)')

    # Labels and title
    ax.set_xlabel('PPVP Disparity (Lower = More Fair)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score (Higher = Better Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('Phase 3: Accuracy vs Fairness Trade-off Curve\nDitto Entity Matching on Compas Dataset',
                fontsize=14, fontweight='bold', pad=15)

    # Add quadrant label for ideal region
    ax.text(0.02, 0.98, 'Ideal Region:\nHigh Accuracy\nHigh Fairness',
           transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='darkgreen'),
           fontsize=10, fontweight='bold')

    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def plot_alpha_sensitivity(results_df, save_path=None, show_plot=True):
    """
    Generate detailed alpha sensitivity analysis showing how each metric varies with alpha.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Alpha Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: F1 Score vs Alpha
    ax1 = axes[0, 0]
    ax1.plot(results_df['alpha'], results_df['f1_score'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Alpha (α)', fontweight='bold')
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('Accuracy vs Alpha', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=results_df['f1_score'].max(), color='g', linestyle='--', alpha=0.5, label='Maximum')
    ax1.legend()

    # Plot 2: PPVP Disparity vs Alpha
    ax2 = axes[0, 1]
    ax2.plot(results_df['alpha'], results_df['ppvp_disparity'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Alpha (α)', fontweight='bold')
    ax2.set_ylabel('PPVP Disparity', fontweight='bold')
    ax2.set_title('Fairness Disparity vs Alpha', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=results_df['ppvp_disparity'].min(), color='g', linestyle='--', alpha=0.5, label='Minimum (Best)')
    ax2.legend()

    # Plot 3: Fairness Rate vs Alpha
    ax3 = axes[1, 0]
    ax3.plot(results_df['alpha'], results_df['fairness_rate'] * 100, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Alpha (α)', fontweight='bold')
    ax3.set_ylabel('Fairness Rate (%)', fontweight='bold')
    ax3.set_title('Fair Groups Percentage vs Alpha', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=100, color='darkgreen', linestyle='--', alpha=0.5, label='Perfect Fairness')
    ax3.legend()

    # Plot 4: Training Time vs Alpha
    ax4 = axes[1, 1]
    if 'training_time_seconds' in results_df.columns:
        ax4.plot(results_df['alpha'], results_df['training_time_seconds'], 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Alpha (α)', fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax4.set_title('Computational Cost vs Alpha', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alpha sensitivity plot saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def plot_loss_curves(results_df, save_path=None, show_plot=True):
    """
    Generate loss curves per alpha:
    X axis = alpha value, Y axis = loss.
    Shows two bars/lines: accuracy loss and fairness loss from final metrics.
    
    Note: For per-step loss curves, the training logs would need to be parsed.
    This provides a summary view from the results CSV.
    """
    if 'ppvp_disparity' not in results_df.columns:
        print("Skipping loss curves — no disparity data in results.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    alphas = results_df['alpha'].values
    f1_scores = results_df['f1_score'].values
    disparities = results_df['ppvp_disparity'].values

    # Dual axis: F1 on left, disparity on right
    color1 = '#2196F3'
    color2 = '#FF5722'

    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', color=color1, fontsize=13, fontweight='bold')
    line1 = ax.plot(alphas, f1_scores, 'o-', color=color1, linewidth=2.5, markersize=10, label='F1 Score')
    ax.tick_params(axis='y', labelcolor=color1)

    ax2 = ax.twinx()
    ax2.set_ylabel('PPVP Disparity', color=color2, fontsize=13, fontweight='bold')
    line2 = ax2.plot(alphas, disparities, 's--', color=color2, linewidth=2.5, markersize=10, label='PPVP Disparity')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=11)

    ax.set_title('Accuracy vs Fairness Trade-off by Alpha\n(F1 Score & PPVP Disparity)',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss/trade-off curve saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def generate_statistical_summary(results_df, save_path=None):
    """
    Generate statistical analysis and summary table.
    """
    summary = {}

    # Baseline (α=0.0) stats
    baseline_rows = results_df[results_df['alpha'] == 0.0]
    if len(baseline_rows) == 0:
        baseline_rows = results_df[results_df['alpha'] == results_df['alpha'].min()]
    baseline = baseline_rows.iloc[0]
    summary['baseline_f1'] = baseline['f1_score']
    summary['baseline_disparity'] = baseline['ppvp_disparity']

    # Best fairness stats
    best_fair = results_df[results_df['alpha'] == results_df['alpha'].max()].iloc[0]
    summary['best_fair_f1'] = best_fair['f1_score']
    summary['best_fair_disparity'] = best_fair['ppvp_disparity']

    # Improvement metrics
    summary['disparity_reduction'] = baseline['ppvp_disparity'] - best_fair['ppvp_disparity']
    if baseline['ppvp_disparity'] > 0:
        summary['disparity_reduction_pct'] = (summary['disparity_reduction'] / baseline['ppvp_disparity']) * 100
    else:
        summary['disparity_reduction_pct'] = 0.0
    summary['f1_cost'] = baseline['f1_score'] - best_fair['f1_score']
    if baseline['f1_score'] > 0:
        summary['f1_cost_pct'] = (summary['f1_cost'] / baseline['f1_score']) * 100
    else:
        summary['f1_cost_pct'] = 0.0

    # Correlation analysis
    summary['alpha_f1_correlation'] = results_df['alpha'].corr(results_df['f1_score'])
    summary['alpha_disparity_correlation'] = results_df['alpha'].corr(results_df['ppvp_disparity'])

    # Optimal alpha (best trade-off)
    acceptable_f1 = baseline['f1_score'] * 0.95
    candidates = results_df[results_df['f1_score'] >= acceptable_f1]

    if len(candidates) > 0:
        optimal = candidates.loc[candidates['ppvp_disparity'].idxmin()]
        summary['optimal_alpha'] = optimal['alpha']
        summary['optimal_f1'] = optimal['f1_score']
        summary['optimal_disparity'] = optimal['ppvp_disparity']
        summary['optimal_fairness_rate'] = optimal['fairness_rate']
    else:
        summary['optimal_alpha'] = None

    # Print summary
    print("\n" + "="*80)
    print(" STATISTICAL SUMMARY")
    print("="*80)
    print(f"\nBaseline (α={baseline['alpha']:.1f}):")
    print(f"  F1 Score: {summary['baseline_f1']:.4f}")
    print(f"  PPVP Disparity: {summary['baseline_disparity']:.4f}")

    print(f"\nHighest Alpha (α={best_fair['alpha']:.1f}):")
    print(f"  F1 Score: {summary['best_fair_f1']:.4f} (Δ: {summary['f1_cost']:.4f}, {summary['f1_cost_pct']:.1f}%)")
    print(f"  PPVP Disparity: {summary['best_fair_disparity']:.4f} (Δ: {summary['disparity_reduction']:.4f}, {summary['disparity_reduction_pct']:.1f}%)")

    if summary['optimal_alpha'] is not None:
        print(f"\nRecommended Optimal (α={summary['optimal_alpha']:.1f}):")
        print(f"  F1 Score: {summary['optimal_f1']:.4f}")
        print(f"  PPVP Disparity: {summary['optimal_disparity']:.4f}")
        print(f"  Fairness Rate: {summary['optimal_fairness_rate']:.2%}")

    print(f"\nCorrelations:")
    print(f"  Alpha ↔ F1: {summary['alpha_f1_correlation']:.3f}")
    print(f"  Alpha ↔ Disparity: {summary['alpha_disparity_correlation']:.3f}")
    print("="*80 + "\n")

    # Save to file if requested
    if save_path:
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(save_path, index=False)
        print(f"Statistical summary saved: {save_path}")

    return summary


def create_comprehensive_report(results_df, output_dir):
    """
    Generate complete analysis report with all visualizations.

    Args:
        results_df: DataFrame with phase3 results
        output_dir: Directory to save all outputs (typically run_dir/visuals/)
    """
    print("\n" + "="*80)
    print(" GENERATING COMPREHENSIVE REPORT")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pareto Frontier
    print("Generating Pareto frontier plot...")
    pareto_path = output_dir / 'phase3_pareto_frontier.png'
    plot_pareto_frontier(results_df, save_path=pareto_path, show_plot=False)

    # 2. Alpha Sensitivity
    print("Generating alpha sensitivity analysis...")
    sensitivity_path = output_dir / 'phase3_alpha_sensitivity.png'
    plot_alpha_sensitivity(results_df, save_path=sensitivity_path, show_plot=False)

    # 3. Trade-off / Loss Curve
    print("Generating trade-off curve...")
    tradeoff_path = output_dir / 'phase3_tradeoff_curve.png'
    plot_loss_curves(results_df, save_path=tradeoff_path, show_plot=False)

    # 4. Statistical Summary
    print("Generating statistical summary...")
    summary_path = output_dir / 'phase3_statistical_summary.csv'
    summary = generate_statistical_summary(results_df, save_path=summary_path)

    # 5. Detailed results table
    detailed_path = output_dir / 'phase3_detailed_results.csv'
    results_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved: {detailed_path}")

    print("\n" + "="*80)
    print(" REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")
    print("="*80 + "\n")


def find_latest_run_dir(base_dir):
    """Find the most recent run folder under base_dir by sorting folder names."""
    base = Path(base_dir)
    if not base.exists():
        return None
    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name[0:4].isdigit()]
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]


def main():
    """Main entry point for visualization"""
    parser = argparse.ArgumentParser(description='Phase 3: Results Visualization')

    parser.add_argument('--run_dir', type=str, default=None,
                       help='Path to a specific run folder (e.g., results/phase3/2026-03-05_18-29-25-123)')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to phase3 results CSV file (legacy mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations (default: run_dir/visuals/)')
    parser.add_argument('--show_plots', action='store_true',
                       help='Display plots interactively')

    args = parser.parse_args()

    # --- Determine results file and output directory ---
    if args.run_dir:
        run_dir = Path(args.run_dir)
        results_path = run_dir / 'phase3_loss_curve_results.csv'
        output_dir = run_dir / 'visuals'
    elif args.results_file:
        results_path = Path(args.results_file)
        output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / 'visualizations'
    else:
        # Auto-detect: find the latest run folder
        default_base = Path(__file__).parent / '..' / 'results' / 'phase3'
        latest = find_latest_run_dir(default_base)
        if latest and (latest / 'phase3_loss_curve_results.csv').exists():
            results_path = latest / 'phase3_loss_curve_results.csv'
            output_dir = latest / 'visuals'
            print(f"Auto-detected latest run: {latest.name}")
        else:
            # Fallback to old flat layout
            results_path = default_base / 'phase3_loss_curve_results.csv'
            output_dir = default_base / 'visualizations'

    if args.output_dir:
        output_dir = Path(args.output_dir)

    # Check if results file exists
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("\nPlease run phase3_experiment.py first to generate results,")
        print("or specify the run directory with --run_dir")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_path}")
    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} experimental results\n")

    # Generate comprehensive report
    create_comprehensive_report(results_df, output_dir)

    # Optionally show plots
    if args.show_plots:
        matplotlib.use('TkAgg')
        print("\nDisplaying plots...")
        plot_pareto_frontier(results_df, show_plot=True)
        plot_alpha_sensitivity(results_df, show_plot=True)


if __name__ == "__main__":
    main()
