---
jupytext:
  formats: md:myst
  notebook_metadata_filter: all,-language_info
  split_at_heading: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.11.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


```{code-cell} python
:tags: [remove-cell]
import numpy as np
import pandas as pd
# Some configurations to "beautify" plots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Libre Franklin"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
```

# Making predictions about quality


First, let's load the ABIDE dataset, and apply the site-wise normalization.

```{code-cell} python
from mriqc_learn.datasets import load_dataset
from mriqc_learn.models.preprocess import SiteRobustScaler

(train_x, train_y), _ = load_dataset(split_strategy="none")
train_x = train_x.drop(columns=[
    "size_x",
    "size_y",
    "size_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
])
numeric_columns = train_x.columns.tolist()
train_x["site"] = train_y.site

# Harmonize between sites
scaled_x = SiteRobustScaler(unit_variance=True).fit_transform(train_x)
```

This time, we are going to *massage* the target labels, adapting it to a binary classification problem.
Since we have three classes, we are going to merge the "*doubtful*" label into the "*discard*" label:

```{code-cell} python
train_y = train_y[["rater_3"]].values.squeeze().astype(int)
print(f"Discard={100 * (train_y == -1).sum() / len(train_y):.2f}%.")
print(f"Doubtful={100 * (train_y == 0).sum() / len(train_y):.2f}%.")
print(f"Accept={100 * (train_y == 1).sum() / len(train_y):.2f}%.")
train_y += 1
train_y = (~train_y.astype(bool)).astype(int)
print(100 * train_y.sum() / len(train_y))  # Let's double check we still have 15% discard
```

Because we already had 1.5% of *doubtful* cases, our final dataset will remain quite imbalanced (\~85% vs. \~15%).

## Setting up a classifier

Let's bundle together the dimensionality reduction and a random forests classifier within a `Pipeline` with *scikit-learn*.
The first entry of our pipeline will drop the `site` column of our data table.

```{code-cell} python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from mriqc_learn.models.preprocess import DropColumns

model = Pipeline([
    ("drop_site", DropColumns(drop=["site"])),
    ("pca", PCA(n_components=4)),
    (
        "rfc", 
        RFC(
          bootstrap=True,
          class_weight=None,
          criterion="gini",
          max_depth=10,
          max_features="sqrt",
          max_leaf_nodes=None,
          min_impurity_decrease=0.0,
          min_samples_leaf=10,
          min_samples_split=10,
          min_weight_fraction_leaf=0.0,
          n_estimators=400,
          oob_score=True,
      ),
    ),
])
```

## Evaluating performance with cross-validation

And we are going to obtain a cross-validated estimate of performance.
By leaving one whole site out for validation at a time, we can get a sense of how this classifier may perform on unseen, new data.

```{code-cell} python
from mriqc_learn.model_selection import split
from sklearn.model_selection import cross_val_score

# Define a splitting strategy
outer_cv = split.LeavePSitesOut(1, robust=True)

cv_score = cross_val_score(
    model,
    X=scaled_x,
    y=train_y,
    cv=outer_cv,
    scoring="roc_auc",
)
print(f"AUC (cross-validated, {len(cv_score)} folds): {cv_score.mean()} ± {cv_score.std()}")
```

## Evaluating performance on a left-out dataset

We start by loading an additional dataset containing the IQMs extracted from [OpenNeuro's *ds0000030*](https://openneuro.org/datasets/ds000030/).
We will then drop some columns we won't need, and discard a few datapoints for which an aliasing ghost idiosyncratic of this dataset is found.
We do this because our simple models will not correctly identify this artifact.

```{code-cell} python
(test_x, test_y), _ = load_dataset(
    "ds030",
    split_strategy="none",
)
test_x = test_x.drop(columns=[
    "size_x",
    "size_y",
    "size_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
])
test_x["site"] = test_y.site

# Discard datasets with ghost
has_ghost = test_y.has_ghost.values.astype(bool)
test_y = test_y[~has_ghost]

# Harmonize between sites
scaled_test_x = SiteRobustScaler(unit_variance=True).fit_transform(
    test_x[~has_ghost]
)
```

Prepare the targets in the same way:

```{code-cell} python
test_y = test_y[["rater_1"]].values.squeeze().astype(int)
print(f"Discard={100 * (test_y == -1).sum() / len(test_y):.2f}%.")
print(f"Doubtful={100 * (test_y == 0).sum() / len(test_y):.2f}%.")
print(f"Accept={100 * (test_y == 1).sum() / len(test_y):.2f}%.")
test_y += 1
test_y = (~test_y.astype(bool)).astype(int)
print(100 * test_y.sum() / len(test_y))
```

Now we train our classifier in all available data:

```{code-cell} python
classifier = model.fit(X=train_x, y=train_y)
```

And we can make predictions about the new data:

```{code-cell} python
predicted_y = classifier.predict(scaled_test_x)
```

And calculate the performance score (AUC):

```{code-cell} python
from sklearn.metrics import roc_auc_score as auc

print(f"AUC on DS030 is {auc(test_y, predicted_y)}.")
```

Okay, that is embarrassing.
We have a model with modest estimated performance (AUC=0.63, on average), but it totally failed when applied on a left-out dataset (AUC=0.5).
An alternative and useful way of understanding the predictions is plotting the confusion matrix:

```{code-cell} python
from sklearn.metrics import classification_report

print(classification_report(test_y, predicted_y, zero_division=0))
```

As we can see, all test samples were predicted as having *good* quality.

## Changing the dimensionality reduction strategy

Let's design a new model using an alternative methodology to dimensionality reduction.
In this case, we will use locally linear embedding:

```{code-cell} python
from sklearn.manifold import LocallyLinearEmbedding as LLE

model_2 = Pipeline([
    ("drop_site", DropColumns(drop=["site"])),
    ("lle", LLE(n_components=3)),
    (
        "rfc", 
        RFC(
          bootstrap=True,
          class_weight=None,
          criterion="gini",
          max_depth=10,
          max_features="sqrt",
          max_leaf_nodes=None,
          min_impurity_decrease=0.0,
          min_samples_leaf=10,
          min_samples_split=10,
          min_weight_fraction_leaf=0.0,
          n_estimators=400,
          oob_score=True,
      ),
    ),
])

cv_score_2 = cross_val_score(
    model_2,
    X=scaled_x,
    y=train_y,
    cv=outer_cv,
    scoring="roc_auc",
)
print(f"AUC (cross-validated, {len(cv_score_2)} folds): {cv_score_2.mean()} ± {cv_score_2.std()}")
```

Now, the cross-validated performance is even worse.
Predictions on the left out dataset are just the same, all samples were predicted as *good*:

```{code-cell} python
pred_y_2 = model_2.fit(X=train_x, y=train_y).predict(scaled_test_x)
print(f"AUC on DS030 is {auc(test_y, pred_y_2)}.")
print(classification_report(test_y, pred_y_2, zero_division=0))
```

## Using the MRIQC classifier

Finally, let's check the internal classifier design of MRIQC.
It can be instantiated with `init_pipeline()`:

```{code-cell} python
from mriqc_learn.models.production import init_pipeline

# Reload the datasets as MRIQC's model will want to see the removed columns
(train_x, sites), _ = load_dataset(split_strategy="none")
train_x["site"] = sites.site
(test_x, sites), _ = load_dataset("ds030", split_strategy="none")
test_x["site"] = sites.site

cv_score_3 = cross_val_score(
    init_pipeline(),
    X=train_x,
    y=train_y,
    cv=outer_cv,
    scoring="roc_auc",
    n_jobs=16,
)
print(f"AUC (cross-validated, {len(cv_score_3)} folds): {cv_score_3.mean()} ± {cv_score_3.std()}")

pred_y_3 = init_pipeline().fit(X=train_x, y=train_y).predict(test_x[~has_ghost])
print(f"AUC on DS030 is {auc(test_y, pred_y_3)}.")
print(classification_report(test_y, pred_y_3, zero_division=0))
```
