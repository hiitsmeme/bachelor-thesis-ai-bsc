# Practical Work in AI - Embedding Propagation for Molecular Target Activity Prediction

## Abstract
The aim of this study is to evaluate if embedding propagation improves the performance of an MLP in one-shot molecular target activity prediction. Molecule data is often sparse and extremely skewed, as positive samples are rare. Therefore, it is necessary to verify whether methods that work on other, more traditional datasets translate well to this special scenario. We compared results of a baseline random forest, an MLP and an MLP with embedding propagation. ROC-AUC and Delta AUC-PR were compared for all three models for ten runs, with a Wilcoxon rank-sum test calculated as a last step for these results. On average, neither ROC-AUC nor Delta AUC-PR was significantly impacted. However, the implementation of embedding propagation seems to stabilize results and model behavior for different tasks. Unfortunately, none of the relationships turned out to be statistically significant, but they offer important first insights. These results highlight the need to adapt methods from different fields to make them more suitable for a molecular context.


## 1. Installation and Dependencies
Clone the repository and install all necessary dependencies:
```bash
pip install -r requirements.txt
```

## 2. General
All experiments outlined in the thesis can be reproduced with the provided code without any modifications. Most of the files are jupyter notebooks, all of them are self-contained. Every file can be run from start to finish to obtain all necesary results for further steps.

**a. Preprocessing**
All features for training the models are extraced from the provided molecule data in compute-features-checkpoint.ipynb. The final data is saved in data_scaled.npz in the data folder, which is then imported at the beginning of every model file.

**b. Model Results**
For each of the three models, metrics are computed and saved in a separate jupyter notebook. In each file, the corresponding model is trained and evaluated. At the end, ROC-AUC and Delta AUC-PR are saved in json files in the metrics subfolder for further processing. Each metric is saved once per run, as an average over tasks, once per task, as an average over runs, and once as averaged over both tasks and runs.

**c. Figures**
All figures that are used in the report were created in the figures notebook. The metrics for each model are loaded from the metrics subfolder and processed into figures.

**d. Statistical Tests**
A Wilcoxon rank-sum test was applied to every combination of two models and for both metrics.

## 3. Code Structure
```bash
├── README.md           
├── data/                       # MUV dataset, computed features for model training
├── metrics/                    # json output, three files per model
├── pretraining_notebooky/           # all files used to get the MLP architecture
├── finetuning_notebooky/            # few-shot setting, get final test results on held out tasks
├── compute_features_checkpoint.ipynb       # MUV data preprocessing, feature extraction
├── figures.ipynb                           # create figures for report based on metrics
├── stat_test.ipynb             # perform Wilcoxon rank-sum test
└── requirements.txt            # Python dependencies
```
