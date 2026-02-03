import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNRegressor

OUTPUT_DIR = "output"
noise = 1e-3
DATASET_FILE = os.path.join("data", "Evaluated_Datasets.csv")

class looptabfm:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _val_proxy(self, y_pseudo):
        """Calculates proxy validation loss based on distance from hard labels."""
        return np.mean(np.minimum(np.abs(y_pseudo - 1), np.abs(y_pseudo + 1)))

    def _prepare_labels(self, y_raw):
        """Encodes labels to {-1, 1}."""
        le = LabelEncoder()
        y_int = le.fit_transform(y_raw)
        return y_int * 2 - 1

    def train_and_evaluate(self, X_tr, y_tr, X_test, y_test, params):
        """Core logic for a single self-training run."""
        ss, n_lab, n_unlab, n_iters, n_all_unlab = params
        # Reproducible randomness
        rng = np.random.RandomState(ss * 100)
        
        # Select Labeled Initial Set
        idxs = np.arange(len(X_tr))
        while True:
            lab_idx = rng.choice(idxs, size=n_lab, replace=False)
            X_init, y_init = X_tr[lab_idx], y_tr[lab_idx]
            # Ensure both classes are present
            if len(np.unique(y_init)) == 2:
                # Add noise to soft labels
                y_init = y_init.astype(float) + rng.normal(0, noise, size=n_lab)
                break
        
        # Select Unlabeled Pool
        pool_idx = np.setdiff1d(idxs, lab_idx)[:n_all_unlab]
        X_pool = X_tr[pool_idx]

        val_best = np.zeros(n_iters + 1)
        tst_best = np.zeros(n_iters + 1)

        # Base Model (Loop-0)
        clf = TabPFNRegressor(n_estimators=2).fit(X_init, y_init)
        y_pool = clf.predict(X_pool)
        
        val_best[0] = self._val_proxy(y_pool)
        initial_preds = np.where(clf.predict(X_test) < 0, -1, 1)
        tst_best[0] = accuracy_score(y_test, initial_preds)

        # Model Update
        for it in range(1, n_iters + 1):
            limit = min(n_unlab * it, n_all_unlab)
            X_train_augmented = np.vstack([X_init, X_pool[:limit]])
            y_train_augmented = np.concatenate([y_init, y_pool[:limit]])
            
            clf = TabPFNRegressor().fit(X_train_augmented, y_train_augmented)
            y_pool = clf.predict(X_pool)

            curr_val = self._val_proxy(y_pool)
            
            # Model Validation
            if curr_val <= val_best[it-1]:
                val_best[it] = curr_val
                test_preds = np.where(clf.predict(X_test) < 0, -1, 1)
                tst_best[it] = accuracy_score(y_test, test_preds)
            else:
                val_best[it] = val_best[it-1]
                tst_best[it] = tst_best[it-1]

        return val_best, tst_best

    def run_full_experiment(self, X, y, dataset_name, **kwargs):
        """Orchestrates multiple splits and parameter combinations."""
        n_tt_splits = kwargs.get('n_tt_splits', 100)
        n_init_lab_list = kwargs.get('n_init_lab_list', [10])
        n_init_unlab_list = kwargs.get('n_init_unlab_list', [10])
        n_iters = kwargs.get('n_iters', 5)
        n_all_unlab = kwargs.get('n_all_unlab', 40)

        shape = (n_tt_splits, len(n_init_lab_list), len(n_init_unlab_list), n_iters + 1)
        accs_val = np.zeros(shape)
        accs_tst = np.zeros(shape)

        for ss in range(n_tt_splits):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.8, random_state=ss)
            
            for i, n_lab in enumerate(n_init_lab_list):
                for j, n_unlab in enumerate(n_init_unlab_list):
                    params = (ss, n_lab, n_unlab, n_iters, n_all_unlab) 
                    val_b, tst_b = self.train_and_evaluate(X_tr, y_tr, X_te, y_te, params)
                    accs_val[ss, i, j, :] = val_b
                    accs_tst[ss, i, j, :] = tst_b

        self._save_results(dataset_name, accs_val, accs_tst, n_init_lab_list, n_init_unlab_list, n_iters)

    def _save_results(self, name, accs_val, accs_tst, lab_list, unlab_list, n_iters):
        """Excel and Plot generation."""
        excel_path = os.path.join(self.output_dir, f"{name}_results.xlsx")
        
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            for i, n_lab in enumerate(lab_list):
                # Process data for storage
                df_tst = pd.DataFrame(accs_tst[:, i, :, :].mean(axis=0), 
                                      index=unlab_list, 
                                      columns=[f"iter_{k}" for k in range(n_iters+1)])
                
                df_tst.to_excel(writer, sheet_name=f"lab_{n_lab}_test")
                self._plot_results(name, n_lab, unlab_list, df_tst, n_iters)

    def _plot_results(self, name, n_lab, unlab_list, df_tst, n_iters):
        plt.figure(figsize=(8, 5))
        for n_unlab in unlab_list:
            plt.plot(range(n_iters + 1), df_tst.loc[n_unlab].values, marker="x", label=f"# of unlabeled = {n_unlab}")

        plt.title(f"{name} (Initial Labeled: {n_lab})")
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{name}_lab_{n_lab}.png"))
        plt.close()


def main():
    try:
        df_ids = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found.")
        return

    exp = looptabfm()

    for _, row in df_ids.iterrows():
        oid = int(row["OpenML ID"])
        print(f"--- Processing OpenML ID: {oid} ---")
        
        X, y_raw = fetch_openml(data_id=oid, as_frame=False, return_X_y=True, parser='auto')
        y = exp._prepare_labels(y_raw)
        
        exp.run_full_experiment(np.array(X), y, dataset_name=str(oid))

if __name__ == "__main__":
    main()