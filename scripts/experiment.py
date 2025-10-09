# scripts/experiment.py

# Note: This experiment matches the evaluation protocol in Zhang et al. (2019)
# and compares performance on the Binary Alphadigits (BA) dataset

import sys
from collections import defaultdict

import graforvfl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from rvfl.model import RVFL

data = scipy.io.loadmat("./data/binaryalphadigs.mat", spmatrix=False)
imgs = data["dat"]
C, N = imgs.shape
H, W = imgs[0, 0].shape

X = np.stack([imgs[i, j].ravel() for i in range(C) for j in range(N)], axis=0)
labels = np.array([[y] * 39 for x in data["classlabels"][0] for y in x])
y = labels.reshape(-1)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = defaultdict(list)

weight_schemes = ["zeros", "uniform", "range"]
for _, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    for w in weight_schemes:
        rvfl_model = RVFL(
            n_hidden=100,
            activation="sigmoid",
            weight_scheme=w,
            seed=42
            )
        rvfl_model.fit(X_train, y_train)

        elm_model = RVFL(
            n_hidden=100,
            activation="sigmoid",
            weight_scheme=w,
            direct_links=False,
            seed=42
            )
        elm_model.fit(X_train, y_train)

        y_hat = rvfl_model.predict_proba(X_test)
        auc_estimator = roc_auc_score(y_test, y_hat, multi_class="ovo")

        results[f"rvfl_{w}"].append(auc_estimator)

        y_hat = elm_model.predict_proba(X_test)
        auc_estimator = roc_auc_score(y_test, y_hat, multi_class="ovo")

        results[f"elm_{w}"].append(auc_estimator)

    grafo_rvfl = graforvfl.RvflClassifier(
        size_hidden=100,
        act_name="sigmoid",
        weight_initializer="random_uniform",
        reg_alpha=None,
        seed=0
    )
    grafo_rvfl.fit(X_train, y_train)

    y_hat = grafo_rvfl.predict_proba(X_test)
    auc_estimator = roc_auc_score(y_test, y_hat, multi_class="ovo")

    results["graforvfl"].append(auc_estimator)

print(results)
fig, ax = plt.subplots()
sns.violinplot(results, ax=ax, inner="quart")
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Estimator Type")
ax.set_title("Estimator Performances on Binary Alphadigits Dataset, \
10 fold CV\n 100 hidden nodes")
ax.hlines(0.92, color="red", xmin=0, xmax=6, label="Plain RVFL Zhang et al.", ls="--")
ax.hlines(0.95, color="gray", xmin=0, xmax=6, label="SP-RVFL Zhang et al.", ls="--")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.legend()
ax.set_ylim(0.4, 1.0)
fig.tight_layout()
fig.savefig("figs/test_oct_2.png", dpi=300)
