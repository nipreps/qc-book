---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: R
  language: R
  name: ir
---

# Some title

Let's see a sensitivity analysis of anova:

```{code-cell} R
library(pwr)

pwr.anova.test(k=2, n=600, sig.level=0.005, power=0.8)
```

