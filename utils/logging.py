import numpy as np
import json
import torch

def log_results(avg_results, all_results, filepath_normal, filepath_per_run, filepath_per_task):
    # compute avg and sd over five runs
    final_auc_pr = np.mean(np.array(avg_results["Delta-AUC-PR"]))
    final_roc_auc = np.mean(np.array(avg_results["ROC-AUC"]))
    final_sd_auc_pr = np.std(np.array(avg_results["Delta-AUC-PR"]))
    final_sd_roc_auc = np.std(np.array(avg_results["ROC-AUC"]))

    # save in json convertible format
    final_results = [{
        "Delta-AUC-PR": final_auc_pr,
        "ROC-AUC": final_roc_auc,
        "Sd-Delta-AUC-PR": final_sd_auc_pr,
        "Sd-ROC-AUC": final_sd_roc_auc
    }]

    # save to json file
    with open(filepath_normal, "w") as f:
        json.dump(final_results, f, indent=4)


    all_results = {
        key: [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
        for key, value in all_results.items()
    }


    # calculate mean and sd for five runs but don't average over tasks
    avg_auc_pr_per_task = np.mean(np.array(all_results["Delta-AUC-PR"]), axis=0)
    sd_auc_pr_per_task = np.std(np.array(all_results["Delta-AUC-PR"]), axis=0)
    avg_roc_auc_per_task = np.mean(np.array(all_results["ROC-AUC"]), axis=0)
    sd_roc_auc_per_task = np.std(np.array(all_results["ROC-AUC"]), axis=0)

    # save in json convertible format
    final_results_per_task = [{
        "Delta-AUC-PR per task": list(avg_auc_pr_per_task),
        "ROC-AUC per task": list(avg_roc_auc_per_task),
        "Sd-Delta-AUC-PR per task": list(sd_auc_pr_per_task),
        "Sd-ROC-AUC per task": list(sd_roc_auc_per_task)
    }]
    final_results_per_task

    # write to json file
    with open(filepath_per_task, "w") as f:
        json.dump(final_results_per_task, f, indent=4)



    # calculate mean and sd for five runs but do average over tasks
    avg_auc_pr_per_run = np.mean(np.array(all_results["Delta-AUC-PR"]), axis=1)
    sd_auc_pr_per_run = np.std(np.array(all_results["Delta-AUC-PR"]), axis=1)

    avg_roc_auc_per_run = np.mean(np.array(all_results["ROC-AUC"]), axis=1)
    sd_roc_auc_per_run = np.std(np.array(all_results["ROC-AUC"]), axis=1)

    # save in json convertible format
    final_results_per_run = [{
        "Delta-AUC-PR per run": list(avg_auc_pr_per_run),
        "ROC-AUC per run": list(avg_roc_auc_per_run),
        "Sd-Delta-AUC-PR per run": list(sd_auc_pr_per_run),
        "Sd-ROC-AUC per run": list(sd_roc_auc_per_run)
    }]

    # write to json file
    with open(filepath_per_run, "w") as f:
        json.dump(final_results_per_run, f, indent=4)


def log_results_nn(avg_results, all_results, filepath_normal, filepath_per_run, filepath_per_task):
    # compute avg and sd over five runs
    final_auc_pr = np.mean(np.array(avg_results["Delta-AUC-PR"]))
    final_roc_auc = np.mean(np.array(avg_results["ROC-AUC"]))
    final_sd_auc_pr = np.std(np.array(avg_results["Delta-AUC-PR"]))
    final_sd_roc_auc = np.std(np.array(avg_results["ROC-AUC"]))

    # save in json convertible format
    final_results = [{
        "Delta-AUC-PR": float(final_auc_pr),
        "ROC-AUC": float(final_roc_auc),
        "Sd-Delta-AUC-PR": float(final_sd_auc_pr),
        "Sd-ROC-AUC": float(final_sd_roc_auc)
    }]

    # save to json file
    with open(filepath_normal, "w") as f:
        json.dump(final_results, f, indent=4)


    all_results = {
        key: [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
        for key, value in all_results.items()
    }


    # calculate mean and sd for five runs but don't average over tasks
    avg_auc_pr_per_task = np.mean(np.array(all_results["Delta-AUC-PR"]), axis=0)
    sd_auc_pr_per_task = np.std(np.array(all_results["Delta-AUC-PR"]), axis=0)

    avg_roc_auc_per_task = np.mean(np.array(all_results["ROC-AUC"]), axis=0)
    sd_roc_auc_per_task = np.std(np.array(all_results["ROC-AUC"]), axis=0)

    # save in json convertible format
    final_results_per_task = [{
        "Delta-AUC-PR per task": list(avg_auc_pr_per_task),
        "ROC-AUC per task": list(avg_roc_auc_per_task),
        "Sd-Delta-AUC-PR per task": list(sd_auc_pr_per_task),
        "Sd-ROC-AUC per task": list(sd_roc_auc_per_task)
    }]

    # write to json file
    with open(filepath_per_task, "w") as f:
        json.dump(final_results_per_task, f, indent=4)



    # calculate mean and sd for five runs but do average over tasks
    avg_auc_pr_per_run = np.mean(np.array(all_results["Delta-AUC-PR"]), axis=1)
    sd_auc_pr_per_run = np.std(np.array(all_results["Delta-AUC-PR"]), axis=1)

    avg_roc_auc_per_run = np.mean(np.array(all_results["ROC-AUC"]), axis=1)
    sd_roc_auc_per_run = np.std(np.array(all_results["ROC-AUC"]), axis=1)

    # save in json convertible format
    final_results_per_run = [{
        "Delta-AUC-PR per run": list(avg_auc_pr_per_run),
        "ROC-AUC per run": list(avg_roc_auc_per_run),
        "Sd-Delta-AUC-PR per run": list(sd_auc_pr_per_run),
        "Sd-ROC-AUC per run": list(sd_roc_auc_per_run)
    }]

    # write to json file
    with open(filepath_per_run, "w") as f:
        json.dump(final_results_per_run, f, indent=4)


def save_best_model(model, avg_pr_auc, best_score, file_path):
    # Check if the current average PR AUC is better than the best recorded score
    if avg_pr_auc > best_score:
        torch.save(model.state_dict(), file_path)
        print(f"Saved new best model with avg PR AUC: {avg_pr_auc:.4f}")
        return True
    else:
        print(f"Current avg PR AUC: {avg_pr_auc:.4f} did not improve over best score: {best_score:.4f}")
        return False