---
jupytext:
  formats: md:myst
  notebook_metadata_filter: all,-language_info
  split_at_heading: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.11
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} python
:tags: [remove-cell]
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Libre Franklin"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# Load the data as we did previously
from mriqc_learn.datasets import load_dataset
(train_x, train_y), _ = load_dataset(split_strategy="none")
```

# Target labels: manual ratings by humans
Let's go further in exploring this effect, by now focusing our attention on the dataset targets (`train_y`).

```{code-cell} python
from mriqc_learn.viz import ratings
ratings.raters_variability_plot(train_y);
```

