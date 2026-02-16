#!/usr/bin/env python3
"""
Phase 1: Fair Entity Matching Analysis
Focus: Ditto model on Compas (NoFlyCompass) dataset with PPVP measure only

This script reproduces fairness analysis following the original run_example.py pattern
with focus on single model/measure combination using the full dataset.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add fairEM to path
sys.path.append('fair_entity_matching/fairEM')

import FairEM as fem
import workloads as wl


def save_pandas_csv_if_not_exists(dataframe, outname, outdir):
    """Save results following original pattern"""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fullname = os.path.join(outdir, outname)
    dataframe.to_csv(fullname, index=False)
    print(f"Results saved to: {fullname}")




def run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, test_file,
                     single_fairness=True, k_combinations=1, delimiter=','):
    """Create workload following original pattern"""
    predictions = pd.read_csv("fair_entity_matching/fairEM/data/" + model + "/" + dataset + "/preds.csv").values.tolist()
    test_file_path = "fair_entity_matching/fairEM/data/DeepMatcher/" + dataset + "/" + test_file

    workload = wl.Workload(pd.read_csv(test_file_path),
                           sens_att_left=left_sens_attribute, sens_att_right=right_sens_attribute,
                           prediction=predictions, label_column="label",
                           multiple_sens_attr=True, delimiter=delimiter,
                           single_fairness=single_fairness, k_combinations=k_combinations)
    return [workload]


def single_fairness_ppvp_only(model, dataset, left_sens_attribute, right_sens_attribute, threshold,
                              single_fairness=True, test_file="test.csv"):
    """
    Run single fairness analysis with PPVP measure only (following original pattern)
    """
    print(f"Running fairness analysis for {model} on {dataset} dataset")
    print(f"Threshold: {threshold}, Single fairness: {single_fairness}")

    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, test_file,
                                 single_fairness=single_fairness)

    fairEM = fem.FairEM(model, workloads, alpha=0.05, full_workload_test=test_file, threshold=threshold,
                        single_fairness=single_fairness)

    # Focus on PPVP only
    measures = ["positive_predictive_value_parity"]

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        attribute_names.append(curr_attr_name)

    print(f"Subgroups being analyzed: {attribute_names}")

    df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair", "counts"])

    aggregate = "distribution"
    for measure in measures:
        print(f"Processing measure: {measure}")
        temp_df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair", "counts"])
        is_fair, counts = fairEM.is_fair(measure, aggregate)
        temp_df["measure"] = [measure] * len(is_fair)
        temp_df["sens_attr"] = attribute_names
        temp_df["is_fair"] = is_fair
        temp_df["counts"] = counts
        df = pd.concat([df, temp_df], ignore_index=True)

    # Save results
    save_pandas_csv_if_not_exists(dataframe=df, outname=model + "_ppvp_results_single_fairness.csv",
                                  outdir="phase1_results/" + dataset + "/")

    return df


def main():
    """Main execution function following original run_example.py pattern"""
    print("=== Phase 1: Ditto + Compas + PPVP Analysis ===")
    print("Following original research framework with full dataset")

    # Run single fairness analysis with PPVP measure only (following original pattern)
    results = single_fairness_ppvp_only(model="Ditto",
                                       dataset="Compas",
                                       left_sens_attribute="left_Ethnic_Code_Text",
                                       right_sens_attribute="right_Ethnic_Code_Text",
                                       threshold=0.1,
                                       single_fairness=True,
                                       test_file="test.csv")

    # Display results
    print(f"\n=== Phase 1 Results ===")
    print(f"Model: Ditto")
    print(f"Dataset: Compas (full dataset)")
    print(f"Measure: Positive Predictive Value Parity (PPVP)")
    print(f"Threshold: 0.1")
    print(f"Number of subgroups analyzed: {len(results)}")

    print("\nFairness Analysis Results:")
    for _, row in results.iterrows():
        status = "FAIR" if row['is_fair'] else "UNFAIR"
        print(f"  {row['sens_attr']}: {status} (count: {row['counts']})")

    # Summary statistics
    fair_count = sum(results['is_fair'])
    unfair_count = len(results) - fair_count
    print(f"\nSummary:")
    print(f"- Fair subgroups: {fair_count}")
    print(f"- Unfair subgroups: {unfair_count}")
    print(f"- Fairness rate: {fair_count/len(results)*100:.1f}%")

    print(f"\n=== Phase 1 Complete ===")
    print("Results successfully reproduced using original research framework!")


if __name__ == "__main__":
    main()