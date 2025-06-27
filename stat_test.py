import json
import numpy as np
from scipy import stats
import pandas as pd

def load_scores_from_json(file_path, metric_key):
    # Open the JSON file and parse the data
    with open(file_path, "r") as f:
        data = json.load(f)
    # Assuming the JSON data is a list with one dictionary entry
    return np.array(data[0][metric_key])

# File paths for the two models' JSON files
baseline_file = "./metrics/metrics_baseline_ft2_per_run.json"
mlp_file = "./metrics/vanilla_ft2_per_run.json"
mlp_ep_file = "./metrics/ep_ft2_per_run.json"


def perform_tests(metric):
    # Load the scores from each file
    mlp_scores = load_scores_from_json(mlp_file, metric)
    mlp_ep_scores = load_scores_from_json(mlp_ep_file, metric)
    baseline_scores = load_scores_from_json(baseline_file, metric)

    print("Baseline Scores:", baseline_scores)
    print("MLP Scores:", mlp_scores)
    print("MLP-EP Scores:", mlp_ep_scores)

    # Perform the Wilcoxon rank-sum test
    statistic_base_ep, p_value_base_ep = stats.ranksums(baseline_scores, mlp_ep_scores)
    statistic_base_mlp, p_value_base_mlp = stats.ranksums(baseline_scores, mlp_scores)
    statistic_mlp_ep, p_value_mlp_ep = stats.ranksums(mlp_scores, mlp_ep_scores)

    data = {
        "Comparison": [
            "baseline vs mlp_ep",
            "baseline vs mlp",
            "mlp vs mlp_ep"
        ],
        "Statistic": [
            statistic_base_ep,
            statistic_base_mlp,
            statistic_mlp_ep
        ],
        "p_value": [
            p_value_base_ep,
            p_value_base_mlp,
            p_value_mlp_ep
        ]
    }

    df = pd.DataFrame(data)
    return df

# tests for ROC-AUC
df_roc_auc = perform_tests("ROC-AUC per run")
print(df_roc_auc)
# tests for Delta AUC-PR
df_delta_auc_pr = perform_tests("Delta-AUC-PR per run")
print(df_delta_auc_pr)

