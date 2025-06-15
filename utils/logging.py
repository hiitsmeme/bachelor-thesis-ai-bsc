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


def log_results_nn_ft(all_results, filepath_normal, filepath_per_run, filepath_per_task):
    # rows: runs; cols: tasks
    delta_auc_pr_matrix = np.array(all_results["Delta-AUC-PR"]).reshape(4,-1).T
    roc_auc_matrix = np.array(all_results["ROC-AUC"]).reshape(4,-1).T
    # -------- Delta AUC PR ---------- # 
    # compute mean, sd per tasks
    delta_auc_pr_mean_tasks = np.mean(delta_auc_pr_matrix, axis=0)
    delta_auc_pr_sd_tasks = np.std(delta_auc_pr_matrix, axis=0)

    # compute mean, sd per runs
    delta_auc_pr_mean_runs = np.mean(delta_auc_pr_matrix, axis=1)
    delta_auc_pr_sd_runs = np.std(delta_auc_pr_matrix, axis=1)

    # compute mean, sd overall
    delta_auc_pr_mean_overall = np.mean(delta_auc_pr_matrix)
    delta_auc_pr_sd_overall = np.std(delta_auc_pr_matrix)

    # -------- ROC AUC ---------- # 
    # compute mean, sd per tasks
    roc_auc_mean_tasks = np.mean(roc_auc_matrix, axis=0)
    roc_auc_sd_tasks = np.std(roc_auc_matrix, axis=0)

    # compute mean, sd per runs
    roc_auc_mean_runs = np.mean(roc_auc_matrix, axis=1)
    roc_auc_sd_runs = np.std(roc_auc_matrix, axis=1)

    # compute mean, sd overall
    roc_auc_mean_overall = np.mean(roc_auc_matrix)
    rpc_auc_sd_overall = np.std(roc_auc_matrix)

    # save in json convertible format
    final_results = [{
        "Delta-AUC-PR": float(delta_auc_pr_mean_overall),
        "ROC-AUC": float(roc_auc_mean_overall),
        "Sd-Delta-AUC-PR": float(delta_auc_pr_sd_overall),
        "Sd-ROC-AUC": float(rpc_auc_sd_overall)
    }]

    # save to json file
    with open(filepath_normal, "w") as f:
        json.dump(final_results, f, indent=4)

    # save in json convertible format
    final_results_per_task = [{
        "Delta-AUC-PR per task": delta_auc_pr_mean_tasks.tolist(),
        "ROC-AUC per task": roc_auc_mean_tasks.tolist(),
        "Sd-Delta-AUC-PR per task": delta_auc_pr_sd_tasks.tolist(),
        "Sd-ROC-AUC per task": roc_auc_sd_tasks.tolist()
    }]

    # write to json file
    with open(filepath_per_task, "w") as f:
        json.dump(final_results_per_task, f, indent=4)


    # save in json convertible format
    final_results_per_run = [{
        "Delta-AUC-PR per run": delta_auc_pr_mean_runs.tolist(),
        "ROC-AUC per run": roc_auc_mean_runs.tolist(),
        "Sd-Delta-AUC-PR per run": delta_auc_pr_sd_runs.tolist(),
        "Sd-ROC-AUC per run": roc_auc_sd_runs.tolist()
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