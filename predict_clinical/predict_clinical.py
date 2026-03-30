import os
import pandas as pd
import numpy as np
import gc
import sys
from typing import Iterable
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

############## Merging melted table with embedings metadata
def concat_metadata(emb_dir: str, train: bool):
    """ 
    Args:
        emb_dir (str):  path to directory with embeddings and metadata
        train (bool):   if True, train split metadata files are chosen
                        else test split metadata are chosen
    Returns: 
        full_emb_metadata (pd.DataFrame):   all chosen metadata files concatenated
    """
    parts = []
    for file in os.listdir(emb_dir):
        file = os.path.join(emb_dir,file)
        if file.endswith(".csv"):
            if train==True and 'test' in file:
                continue
            elif train==False and 'train' in file:
                continue
            part = pd.read_csv(file)
            embedding_file = file.replace('metadata','embeddings').replace('csv','npy')
            if not os.path.exists(embedding_file):
                    continue
            part["embeddings_file"] = embedding_file
            part["embedding_idx"] = part.index
            parts.append(part)
    full_emb_metadata = pd.concat(parts)
    return full_emb_metadata

def concat_virtues_metadata(emb_dir: str, train: bool):
    """ 
    Args:
        emb_dir (str):  path to directory with embeddings and metadata
        train (bool):   if True, train split metadata files are chosen
                        else test split metadata are chosen
    Returns: 
        full_emb_metadata (pd.DataFrame):   all chosen metadata files concatenated
    """
    datasets = []
    for dataset_dir in os.listdir(emb_dir):
        dataset_dir = os.path.join(emb_dir,dataset_dir)
        for file in os.listdir(dataset_dir):
            file = os.path.join(dataset_dir,file)
            if file.endswith(".csv"):
                if train==True and 'test' in file:
                    continue
                elif train==False and 'train' in file:
                    continue
                dataset = pd.read_csv(file)
                embedding_file = file.replace('metadata','embeddings').replace('csv','npy')
                if not os.path.exists(embedding_file):
                    continue
                dataset["embeddings_file"] = embedding_file
                dataset["embedding_idx"] = dataset.index
                datasets.append(dataset)
    full_emb_metadata = pd.concat(datasets)
    return full_emb_metadata

def merge_metadata_with_melted(emb_metadata: pd.DataFrame, melted_table: pd.DataFrame, meta_table_path: str):
    """ 
    Args:
        emb_metadata (pd.DataFrame):  dataframe with embeddings metadata
        melted_table (pd.DataFrame):  melted table with single feature per single image as row
        meta_table_path (str):        path to destination file for the result
    Returns: 
        merged_table (pd.DataFrame):  melted table with single feature per single embedding as row 
                                    + path to file with embeddings and idx of embedding in that file
                                    (meta table)
    """
    emb_metadata = emb_metadata.copy()
    img_col = "img_path" if "img_path" in emb_metadata.columns else "image_paths" # col with img path in emb_metadata file
    emb_metadata["img_path"] = emb_metadata[img_col].apply(lambda x: x.split("/")[-1])
    merged_table = pd.merge(emb_metadata, melted_table, left_on="img_path", right_on='image_path')
    merged_table = merged_table.drop(columns=['panel','image_path'], errors='ignore')
    os.makedirs(os.path.dirname(meta_table_path), exist_ok=True)
    merged_table.to_csv(meta_table_path)
    return merged_table

############### Helpers for subseting the table

def get_a_subset(meta_table: pd.DataFrame, column: str, value: str):
    return meta_table[meta_table[column]==value]

def get_unique_values(tables: list[pd.DataFrame], column: str) -> set:
    intersection = None
    for table in tables:
        values = set(pd.unique(table[column]))
        if intersection is None:
            intersection = values
        else:
            intersection &= values
    return intersection 

def get_variable_type(table: pd.DataFrame, dataset: str, feature: str):
    """
    Args:
        table (pd.DataFrame): melted table containing features per dataset and their type (eg. categorical, ordinal)
        dataset (str):        which dataset the feature comes from
        feature (str):        which feature's type to return
    Returns:
        (str): type of the feature (eg. categorical, ordinal)
    """
    return table.loc[(table["dataset"] == dataset) & (table["feature"] == feature), "variable_type"].iloc[0]

############### Loading embeddings and labels to variables

def load_labels(feature_df: pd.DataFrame):
    """
    Args:
        feature_df (pd.DataFrame): a subset of meta table with rows per embedding from given dataset for given feature
    Returns:
        (np.array): array with feature values, shape(n_embeddings, )
    """
    return feature_df["feature_value"].values

def load_embeddings_for_training(meta_df: pd.DataFrame, model: str, normalize: bool = False, batch_size: int = 10000):

    """
    Args:
        meta_df (pd.DataFrame): subset of meta table containing rows for one dataset
        normalize (bool):       should embeddigs be normalized with quantile clip and arcisnh
        batch_size (int):       size of embedding batch loaded to RAM before averaging
    Action:
        - takes path to an embedding file and idx in it for each crop in meta_df
        - avereges each embedding by patch (8x8px) grid
        - alterantively applies transformations
    Returns: 
        X (np.array):           array with embeddings coresponding to all crops present in meta_df
                                shape(n_embeddings, ) 
    """

    unique = meta_df.drop_duplicates(subset=["embeddings_file","embedding_idx"])
    
    X_list = []

    for emb_file, group in unique.groupby('embeddings_file'):
        idxs = group['embedding_idx'].values
        emb_array = np.load(emb_file, mmap_mode='r')
        
        # Process in smaller batches to avoid memory spike
        for start in range(0, len(idxs), batch_size):
            print(f"\t batch progress {start} / {len(idxs)}", flush=True)
            batch_idxs = idxs[start:start+batch_size]
            batch_emb = emb_array[batch_idxs]
            batch_emb = np.arcsinh(batch_emb/5)
            if model == "immuvis":
                batch_emb = batch_emb.mean(axis=(2,3))  # average by patch
            X_list.append(batch_emb)
            
    
    X = np.vstack(X_list)
    # if normalize:
    #     print("\t Normalizing the embeddings", flush=True)
    #     q01 = np.quantile(X, 0.01, axis=0)
    #     q99 = np.quantile(X, 0.99, axis=0)
    #     X = np.clip(X, q01, q99)
    #     X = np.arcsinh(X/5)
    #     print("normalized train embeddings")
    #     return X, q01, q99
    return X

def load_embeddings_for_testing(meta_df, model, q01=False, q99=False, normalize=False, batch_size=10000):

    unique = meta_df.drop_duplicates(subset=["embeddings_file","embedding_idx"])
    X_list = []

    for emb_file, group in unique.groupby('embeddings_file'):
        idxs = group['embedding_idx'].values
        emb_array = np.load(emb_file, mmap_mode='r')
        
        # Process in smaller batches to avoid memory spike
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start:start+batch_size]
            batch_emb = emb_array[batch_idxs]
            batch_emb = np.arcsinh(batch_emb/5)
            if model == "immuvis":
                batch_emb = batch_emb.mean(axis=(2,3))  # average by patch
            X_list.append(batch_emb)

    X = np.vstack(X_list)
    # if normalize:
    #     X = np.clip(X, q01, q99)
    #     X = np.arcsinh(X/5)
    #     print("normalized test embeddings")
    return X

############## Prediction metrics and plots

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay


def plot_multiclass_pr_return_avg_prec(y_test, y_pred_proba, plotting_info):
    """
    Plot multi-class Precision-Recall curves (per class + macro-average)
    with iso-F1 curves, given y_test (string labels) and y_pred_proba (DataFrame).
    Returns dict of average precision per class + macro-average.
    """
    
    class_labels = y_pred_proba.columns
    n_classes = len(class_labels)

    # ----- 1. Prepare labels -----
    y_test_bin = label_binarize(y_test, classes=class_labels)
    print(f"ytestbin: {y_test_bin} \n classlabels:{class_labels}")

    if n_classes == 2 and y_test_bin.shape[1] == 1:
        # sklearn binary binarization produces only positive-class column
        # We reconstruct both:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        
    # ----- 2. Compute PR per class -----
    precision = {}
    recall = {}
    average_precision = {}

    for i, label in enumerate(class_labels):
        print(f"i: {i}, label:{label}")
        precision[label], recall[label], _ = precision_recall_curve(
            y_test_bin[:, i], y_pred_proba[label]
        )
        average_precision[label] = float(average_precision_score(
            y_test_bin[:, i], y_pred_proba[label])
        )

    # ----- 3. macro-average -----
    precision["macro"], recall["macro"], _ = precision_recall_curve(
        y_test_bin.ravel(), y_pred_proba.values.ravel()
    )
    average_precision["macro"] = float(average_precision_score(
        y_test_bin, y_pred_proba.values, average="macro")
    )

    # ----- 4. Start plotting -----
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    fig, ax = plt.subplots(figsize=(7, 8))

    # ----- 5. Plot iso-F1 curves -----
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        mask = y >= 0
        ax.plot(x[mask], y[mask], color="gray", alpha=0.2)
        # use middle of the array to avoid IndexError
        idx = len(y[mask]) // 2
        ax.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[mask][idx] + 0.02))


    # ----- 6. macro-average PR curve -----
    display = PrecisionRecallDisplay(
        recall=recall["macro"],
        precision=precision["macro"],
        average_precision=average_precision["macro"],
    )
    display.plot(ax=ax, name="macro-average precision-recall", color="gold")

    # ----- 7. Per-class PR curves -----
    for label, color in zip(class_labels, colors):
        display = PrecisionRecallDisplay(
            recall=recall[label],
            precision=precision[label],
            average_precision=average_precision[label],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {label}", color=color, despine=True)

    # ----- 8. Legend handling -----
    handles, labels_ = display.ax_.get_legend_handles_labels()
    handles.append(ax.lines[0])  # approximate handle for iso-F1 lines
    labels_.append("iso-f1 curves")
    ax.legend(handles=handles, labels=labels_, loc="best")

    dataset, feature, RESULTS_DIR, exp_var = plotting_info
    ax.set_title(f"{dataset} {feature} {exp_var}% of explained variance \n per class Precision-Recall curve")

    return fig, average_precision


############ Prediction logic

def plot_histogram(x, split, pca_treshold, dataset, RESULT_DIR):
    
    x_flat = x.reshape((-1,))

    plt.close()
    hist, ax = plt.subplots()
    ax.hist(x_flat, bins=30)   
    plt.close()
    
    stats = pd.DataFrame({
        "dataset": [dataset],
        "exp_var": [pca_treshold],
        "mean": [x.mean()],
        "median": [np.median(x)],
        "std": [x.std(ddof=1)],
        "cv": [x.std(ddof=1) / (x.mean() + 1e-12)]
    })
        

    reports_dir = os.path.join(RESULTS_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_name = "distribution_table.txt"
    report_path = os.path.join(reports_dir, report_name)
    stats.to_csv(report_path, mode="a", header=False, index=False)
    
    hist_dir = f"{RESULT_DIR}/figures/histograms/{split}"
    os.makedirs(hist_dir, exist_ok=True)
    hist_name = f"{dataset}_{pca_treshold}%expvar_histogram.png"
    hist_path = os.path.join(hist_dir, hist_name)
    hist.savefig(hist_path)
    


def pca(X_train, X_test, dataset, RESULT_DIR):
    """
    returns: 
        dicts storing PCA components of X_train and X_test
        explaining 3 treshods of explained variance
    """
    variance_thresholds = {
        "50": 0.50,
        "75": 0.75,
        "90": 0.90,
        "95": 0.95,
        "99": 0.99,
        "100": None
    }

    pcas = {}
    X_train_pca = {}
    X_test_pca = {}

    for name, thr in variance_thresholds.items():
        pca = PCA(n_components=thr)
        X_train_pca[name] = x_train = pca.fit_transform(X_train)
        X_test_pca[name] = x_test = pca.transform(X_test)   # IMPORTANT: transform, not fit_transform
        plot_histogram(x_train, "train", name, dataset, RESULT_DIR)
        plot_histogram(x_test, "test", name, dataset, RESULT_DIR)
        if name == '95':
            cumulative = pca.explained_variance_ratio_.cumsum()
            # Plot
            plt.figure(figsize=(6, 4))
            plt.plot(cumulative, marker='o')
            plt.xlabel("Number of principal components")
            plt.ylabel("Cumulative explained variance")
            plt.title("Cumulative Explained Variance by PCA Components")
            plt.grid(True)
            fig_name = f"{RESULT_DIR}/figures/explained_var/{dataset}_{name}%.png"
            os.makedirs(os.path.dirname(fig_name), exist_ok=True)
            plt.savefig(fig_name)
        pcas[name] = pca

        print(f"{name}% -> {pca.n_components_} components, {pca.explained_variance_ratio_.sum():.3f} explained variance", flush=True)

    return X_train_pca, X_test_pca

def balance_the_dataset(X_train_pca, X_test_pca, y_train, y_test, train_img_mask, test_img_mask, remove_nan_only=False):
    # === 1. Remove NaNs ===
    train_nan_mask = ~pd.isna(y_train)
    test_nan_mask = ~pd.isna(y_test)

    y_train = y_train[train_nan_mask]
    y_test = y_test[test_nan_mask]
    train_img_mask = train_img_mask[train_nan_mask]
    test_img_mask = test_img_mask[test_nan_mask]

    # === 2. Keep classes >= 5% ===
    if not remove_nan_only:
        unique_classes, counts = np.unique(y_train, return_counts=True)
        freq = counts / counts.sum()
        keep_classes = unique_classes[freq >= 0.05]

        train_keep_mask = np.isin(y_train, keep_classes)
        test_keep_mask = np.isin(y_test, keep_classes)

        y_train = y_train[train_keep_mask]
        y_test = y_test[test_keep_mask]
        train_img_mask = train_img_mask[train_keep_mask]
        test_img_mask = test_img_mask[test_keep_mask]

    # === 3. Shared classes between train/test ===
    shared_classes = np.intersect1d(np.unique(y_train), np.unique(y_test))

    train_shared_mask = np.isin(y_train, shared_classes)
    test_shared_mask = np.isin(y_test, shared_classes)

    y_train = y_train[train_shared_mask]
    y_test = y_test[test_shared_mask]
    train_img_mask = train_img_mask[train_shared_mask]
    test_img_mask = test_img_mask[test_shared_mask]

    # === 4. Apply ALL masks to PCA matrices ===
    X_train_pca_WIP = {}
    X_test_pca_WIP = {}

    for exp_var in X_train_pca:

        if not remove_nan_only:
            x_train = X_train_pca[exp_var][train_nan_mask][train_keep_mask][train_shared_mask]
            x_test =  X_test_pca[exp_var][test_nan_mask][test_keep_mask][test_shared_mask]
        else:
            x_train = X_train_pca[exp_var][train_nan_mask][train_shared_mask]
            x_test =  X_test_pca[exp_var][test_nan_mask][test_shared_mask]

        X_train_pca_WIP[exp_var] = x_train
        X_test_pca_WIP[exp_var] = x_test

    return X_train_pca_WIP, X_test_pca_WIP, y_train, y_test, train_img_mask, test_img_mask




from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

def predict(X_train_pca, X_test_pca, y_train, y_test, train_img_mask, test_img_mask, plotting_info, average_before_regression=True, remove_nan_only=False):
    """
    arguments:
        - X - dict of pca_trashold : embeddings in numpy array
        - y - numpy array of labels
    """

    def predict_single_fold(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        train_img_mask,
        test_img_mask,
        plotting_info,
        average_before_regression=True,
        remove_nan_only=False,
    ):
        classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            class_weight="balanced"
        )

        # Balance / remove NaNs
        X_train_pca, X_test_pca, y_train, y_test, train_img_mask, test_img_mask = (
            balance_the_dataset(
                X_train_pca, X_test_pca,
                y_train, y_test,
                train_img_mask, test_img_mask,
                remove_nan_only=remove_nan_only
            )
        )

        # Image-level labels
        y_train_img = (
            pd.DataFrame({"y": y_train, "img": train_img_mask})
            .groupby("img")["y"]
            .first()
            .values
        )

        y_test_img = (
            pd.DataFrame({"y": y_test, "img": test_img_mask})
            .groupby("img")["y"]
            .first()
            .values
        )

        reports = {}

        for exp_var in X_train_pca:
            x_train = X_train_pca[exp_var]
            x_test  = X_test_pca[exp_var]

            if average_before_regression:
                x_train = (
                    pd.DataFrame(x_train)
                    .assign(img=train_img_mask)
                    .groupby("img")
                    .mean()
                    .values
                )
                x_test = (
                    pd.DataFrame(x_test)
                    .assign(img=test_img_mask)
                    .groupby("img")
                    .mean()
                    .values
                )

            classifier.fit(x_train, y_train_img)
            probs = pd.DataFrame(
                classifier.predict_proba(x_test),
                columns=classifier.classes_
            )

            final_preds = probs.idxmax(axis=1)

            report_dict = classification_report(
                y_test_img,
                final_preds,
                output_dict=True
            )

            # ROC AUC
            if len(classifier.classes_) == 2:
                roc_auc = roc_auc_score(y_test_img, probs.iloc[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test_img, probs, multi_class="ovr"
                )

            report_dict["roc_auc"] = float(roc_auc)

            _, avg_prec = plot_multiclass_pr_return_avg_prec(
                y_test_img, probs, plotting_info + (exp_var,)
            )

            report_dict["average_precision"] = avg_prec

            reports[exp_var] = report_dict

        return reports

    def predict_cv(
        X_pca_all,
        y_all,
        img_mask_all,
        plotting_info,
        n_splits=10,
        n_bootstraps = 1,
        average_before_regression=True,
        remove_nan_only=False,
        random_state=42,
    ):
        # Image-level labels
        img_df = (
            pd.DataFrame({"y": y_all, "img": img_mask_all})
            .groupby("img")["y"]
            .first()
        )

        img_ids = img_df.index.values
        img_labels = img_df.values

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        all_fold_reports = []

        for fold, (train_i, test_i) in enumerate(
            cv.split(img_ids, img_labels, groups=img_ids)
        ):
            print(f"\n===== Fold {fold + 1}/{n_splits} =====")

            train_imgs = img_ids[train_i]
            test_imgs  = img_ids[test_i]

            train_mask = np.isin(img_mask_all, train_imgs)
            test_mask  = np.isin(img_mask_all, test_imgs)

            X_train = {k: v[train_mask] for k, v in X_pca_all.items()}
            X_test  = {k: v[test_mask] for k, v in X_pca_all.items()}

            y_train = y_all[train_mask]
            y_test = y_all[test_mask]

            from sklearn.utils import resample
            for b in range(n_bootstraps):
                
                #X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, replace=True, random_state=b, stratify=y_train, n_samples=int(2/3 * len(X_train)))
                X_train_bootstrap = X_train
                y_train_bootstrap = y_train

                fold_reports = predict_single_fold(
                    X_train_bootstrap,
                    X_test,
                    y_train_bootstrap,
                    y_test,
                    img_mask_all[train_mask],
                    img_mask_all[test_mask],
                    plotting_info,
                    average_before_regression,
                    remove_nan_only,
                )

                # aggregate bootstrap samples

            all_fold_reports.append(fold_reports)

        return all_fold_reports

    def summarize_cv(all_fold_reports):
        agg = defaultdict(lambda: defaultdict(list))

        for fold in all_fold_reports:
            for exp_var, report in fold.items():
                agg[exp_var]["roc_auc"].append(report["roc_auc"])
                agg[exp_var]["avg_prec_macro"].append(
                    report["average_precision"]["macro"]
                )

                for cls, metrics in report.items():
                    if isinstance(metrics, dict) and cls not in ["average_precision"]:
                        for m, v in metrics.items():
                            if m != "support":
                                agg[exp_var][f"{cls}_{m}"].append(v)
                            

        summary = {}

        for exp_var, metrics in agg.items():
            summary[exp_var] = {
                k: (np.mean(v), np.std(v))
                for k, v in metrics.items()
            }

        return summary

    # Merge train + test ONCE
    X_train_pca_WIP, X_test_pca_WIP, y_train, y_test, train_img_mask, test_img_mask = balance_the_dataset(X_train_pca, X_test_pca, y_train, y_test, train_img_mask, test_img_mask, remove_nan_only=remove_nan_only)

    X_all = {k: np.vstack([X_train_pca_WIP[k], X_test_pca_WIP[k]]) for k in X_train_pca_WIP}
    y_all = np.concatenate([y_train, y_test])
    img_mask_all = np.concatenate([train_img_mask, test_img_mask])

    cv_reports = predict_cv(
        X_all,
        y_all,
        img_mask_all,
        plotting_info,
        n_splits=10,
        average_before_regression=True,
    )

    cv_summary = summarize_cv(cv_reports)

    dataset, feature, RESULTS_DIR = plotting_info
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for fold_idx, fold in enumerate(cv_reports):
        for exp_var, report in fold.items():
            for cls, metrics in report.items():
                if isinstance(metrics, dict):
                    for m, v in metrics.items():
                        if m != "support":
                            rows.append([dataset, feature, fold_idx, exp_var, cls, m, v])
                elif cls in ["roc_auc", "average_precision"]:
                    rows.append([dataset, feature, fold_idx, exp_var, cls, "value", metrics])
    df_rows = pd.DataFrame(rows, columns=["dataset", "feature", "fold", "exp_var", "class", "metric", "value"])
    df_rows.to_csv(os.path.join(RESULTS_DIR, "per_fold_results.csv"), mode="a", header=False, index=False)

    return cv_summary


    

def initialize_result_tables(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_per_class_path = os.path.join(RESULTS_DIR, "table_per_class.csv")
    table_per_feature_path = os.path.join(RESULTS_DIR, "table_per_feature.csv")
    per_fold_path = os.path.join(RESULTS_DIR, "per_fold_results.csv")
    pd.DataFrame(columns=["dataset", "feature", "exp_var", "class", "metric", "value"])\
    .to_csv(os.path.join(RESULTS_DIR, "table_per_class.csv"), index=False)

    pd.DataFrame(columns=["dataset", "feature", "exp_var", "metric", "value"])\
    .to_csv(os.path.join(RESULTS_DIR, "table_per_feature.csv"), index=False)

    pd.DataFrame(columns=["dataset", "feature", "fold", "exp_var", "class", "metric", "value"])\
    .to_csv(os.path.join(RESULTS_DIR, "per_fold_results.csv"), index=False)
########### Iterating through datasets and features


FEATURE_FILTER = ['Grade', 'Relapse', 'DX.name', 'ERStatus', 'ERBB2_pos', 'PAM50'] #'simple_histology', 'clinical_type' | 'histology' for jackson-basel (spellings in original clinical features file) 
def filter_features(features: set, filter: list):
    return set(features) & set(filter)

def run(train_meta_table, test_meta_table, var_type_melted_table, RESULTS_DIR, model, normalize=True, average_before_regression=True, remove_nan_only=False):
    
    #datasets = get_unique_values(tables=[train_meta_table,test_meta_table], column="dataset")
    datasets = ['danenberg','cords'] # for testing purposes
    
    initialize_result_tables(RESULTS_DIR)

    results = {}
    
    for dataset in datasets:  
        print(f"Processing {dataset} dataset...", flush=True)
        train_dataset_meta = get_a_subset(meta_table=train_meta_table, column="dataset", value=dataset)
        test_dataset_meta = get_a_subset(meta_table=test_meta_table, column="dataset", value=dataset)
        print(f"dataset size: \n\t training:{len(train_dataset_meta)}\n\t testing: {len(test_dataset_meta)}", flush=True)
        if normalize:
            X_train, q01, q99 = load_embeddings_for_training(train_dataset_meta, model, normalize=normalize)
            X_test = load_embeddings_for_testing(test_dataset_meta, model, q01, q99, normalize=normalize)
        else:
            X_train= load_embeddings_for_training(train_dataset_meta, model, normalize=normalize)
            X_test = load_embeddings_for_testing(test_dataset_meta, model, normalize=normalize)
        X_train_pca, X_test_pca = pca(X_train,X_test, dataset, RESULTS_DIR)
        print("Loaded the embeddings", flush=True)
        features = get_unique_values(tables=[train_dataset_meta], column="feature")
        features = filter_features(features, FEATURE_FILTER)

        if not features:
            print(f"No features found for {dataset}")

        for feature in features:
            print(f"\t Predicting {feature} feature...", flush=True)
            train_feature_meta = get_a_subset(meta_table=train_dataset_meta, column="feature", value=feature)
            test_feature_meta = get_a_subset(meta_table=test_dataset_meta, column="feature", value=feature)
            
            train_img_mask = train_feature_meta["img_path"].values
            test_img_mask = test_feature_meta["img_path"].values
        
            train_labels = load_labels(train_feature_meta)
            test_labels = load_labels(test_feature_meta)        

            print(f"train labels shape: {train_labels.shape}, \n test labels shape: {test_labels.shape}, \n test_img_mask shape: {test_img_mask.shape}", flush=True)
            print(f"train labels: {np.unique(train_labels.astype(str))}, \n test labels: {np.unique(test_labels.astype(str))}", flush=True)
            
            variable_type = get_variable_type(var_type_melted_table, dataset, feature)
            plotting_info = (dataset, feature, RESULTS_DIR)
            
            results = predict(X_train_pca, X_test_pca, train_labels, test_labels, train_img_mask, test_img_mask, plotting_info, average_before_regression=average_before_regression, remove_nan_only=remove_nan_only)
        
            print("Saving the results", flush=True)

            CLASS_EXCLUDE = {"accuracy", "macro avg", "weighted avg", "roc_auc_score"}

            for exp_var, report_dict in results.items():

            
                for key, value in report_dict.items():

                    # -----------------------------
                    # CASE 1: per-class metrics
                    # -----------------------------
                    if isinstance(value, dict) and key not in CLASS_EXCLUDE:

                        exp_var = key
                        for metric, metric_value in value.items():
                            row = pd.DataFrame([[dataset, feature, exp_var, klasa, metric, metric_value]])
                            os.makedirs(RESULTS_DIR, exist_ok=True)
                            table_path = os.path.join(RESULTS_DIR, "table_per_class.csv")
                            row.to_csv(table_path, mode="a", header=None, index=None)

                        continue

                    # -----------------------------
                    # CASE 2: average_precision (special)
                    # -----------------------------
                    if key == "average_precision":
                        for klasa, ap_value in value.items():
                            row = pd.DataFrame([[dataset, feature, exp_var, klasa, "average_precision", ap_value]])
                            table_path = os.path.join(RESULTS_DIR, "table_per_class.csv")
                            row.to_csv(table_path, mode="a", header=None, index=None)

                        continue

                    # -----------------------------
                    # CASE 3: feature-level metrics
                    # -----------------------------
                    row = pd.DataFrame([[dataset, feature, exp_var, key, value]])
                    table_path = os.path.join(RESULTS_DIR, "table_per_feature.csv")
                    row.to_csv(table_path, mode="a", header=None, index=None)

              


            # for pca_treshold in results:

            #     report, prec_recall_plot, roc_auc, average_precision = results[pca_treshold]

            #     fig_dir = os.path.join(RESULTS_DIR, "figures")
            #     os.makedirs(fig_dir, exist_ok=True)
            #     fig_name = f"{dataset}_{feature}_{pca_treshold}%expvar_precission_recall.png"
            #     fig_path = os.path.join(fig_dir, fig_name)
            #     prec_recall_plot.savefig(fig_path)
                    
            #     reports_dir = os.path.join(RESULTS_DIR, "reports")
            #     os.makedirs(reports_dir, exist_ok=True)
            #     report_name = f"{dataset}_{feature}_{pca_treshold}%expvar_report.txt"
            #     report_path = os.path.join(reports_dir, report_name)
            #     with open(report_path, "w") as f:
            #         f.write(report)
            #         f.write("\n")
            #         f.write(f"roc_auc_score: {roc_auc}\n")
            #         f.write(f"average_precision: {average_precision}")
            #     print(f"Finished the {dataset} dataset", flush=True)
                    
        del X_train, X_test, X_train_pca, X_test_pca
        gc.collect()

########## Run
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", 
    required=True,
    choices=["immuvis", "virtues", "ViTS", "ViTM", "ViTL", "DinoVS2"],
    help="Choose the embedding model."
)

parser.add_argument(
    "--average",
    action="store_true",
    help="Average embeddings before regression."
)

parser.add_argument(
    "--no-balance",
    action="store_true",
    help="Remove NaNs only (do not balance dataset)."
)

args = parser.parse_args()

model = args.model
average_before_regression = args.average
remove_nan_only = args.no_balance

print(f"Processing the {model} model with average_emb_to_img = {args.average} and no_balance = {args.no_balance}")
os.chdir("/home/kacpie/toolbox")

var_type_melted_table = pd.read_csv("configs/melted_table_var_types.csv")

if model == 'immuvis':
    EMB_DIR = "/raid_encrypted/immucan/embeddings"
    RESULTS_DIR = "results/immuvis/cross_val/normalized"
    TRAIN_META_TABLE = "configs/immuvis/train_meta_table.csv" #small_train_table.csv" #
    TEST_META_TABLE = "configs/immuvis/test_meta_table.csv" #small_test_table.csv" #
elif model == 'virtues':
    EMB_DIR = "/raid_encrypted/immucan/embeddings_virtues"
    RESULTS_DIR = "results/virtues/cross_val/normalized"
    TRAIN_META_TABLE = "configs/virtues/train_meta_table.csv" #small_train_table.csv" #
    TEST_META_TABLE = "configs/virtues/test_meta_table.csv" #small_test_table.csv" #
elif model.startswith('ViT'):
    size = model[3]
    EMB_DIR = f"/raid_encrypted/immucan/embeddings_ViT/ViT{size}"
    RESULTS_DIR = f"results/ViT{size}/cross_val/normalized"
    TRAIN_META_TABLE = f"configs/ViT{size}/train_meta_table.csv" #small_train_table.csv" #
    TEST_META_TABLE = f"configs/ViT{size}/test_meta_table.csv" #small_test_table.csv" #
elif model.startswith('DinoV'):
    version = model[5:]
    EMB_DIR = f"/raid_encrypted/immucan/embeddings_ViT/DinoV{version}"
    RESULTS_DIR = f"results/DinoV{version}/cross_val/normalized"
    TRAIN_META_TABLE = f"configs/DinoV{version}/train_meta_table.csv" #small_train_table.csv" #
    TEST_META_TABLE = f"configs/DinoV{version}/test_meta_table.csv" #small_test_table.csv" #
MELTED_PATH = "configs/melted_table_images.csv"
melted_table = pd.read_csv(MELTED_PATH)

if average_before_regression:
    RESULTS_DIR = os.path.join(RESULTS_DIR, "averaged_emb", "10fold")

if remove_nan_only:
    RESULTS_DIR = os.path.join(RESULTS_DIR, "no_balance")

os.makedirs(RESULTS_DIR, exist_ok=True)

if os.path.exists(TRAIN_META_TABLE) and os.path.exists(TEST_META_TABLE):
    train_meta_table = pd.read_csv(TRAIN_META_TABLE)
    test_meta_table = pd.read_csv(TEST_META_TABLE)
else:    
    print("Creating the train and test meta tables", flush=True)
    
    if model == 'virtues':
        train_emb_metadata = concat_virtues_metadata(EMB_DIR,train=True)
        test_emb_metadata = concat_virtues_metadata(EMB_DIR,train=False)
    else:
        train_emb_metadata = concat_metadata(EMB_DIR,train=True)
        test_emb_metadata = concat_metadata(EMB_DIR,train=False)
    
    train_meta_table = merge_metadata_with_melted(train_emb_metadata,melted_table,TRAIN_META_TABLE)
    test_meta_table = merge_metadata_with_melted(test_emb_metadata,melted_table,TEST_META_TABLE)
print("Loaded the meta tables", flush=True)

run(train_meta_table, test_meta_table, var_type_melted_table, RESULTS_DIR, model, normalize=False, average_before_regression=average_before_regression, remove_nan_only=remove_nan_only)