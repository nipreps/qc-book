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

# Trying to use R

```{code-cell} R
library(pwr)

pwr.anova.test(k=2, n=600, sig.level=0.005, power=0.8)

# little example of plotting and getting info from niftis:
library(RNifti);  # package for NIfTI image reading; https://github.com/jonclayden/RNifti
# plot S1200_AverageT1w_81x96x81.nii.gz from https://osf.io/6hdxv/

img <- readNifti(fname);    # however to refer to S1200_AverageT1w_81x96x81.nii.gz; url("https://osf.io/6hdxv/") maybe?

# the image values are in a 3d array (single anatomical image, not a timeseries)
# can apply many normal functions
dim(img);   # [1] 81 96 81
max(img);  # [1] 1374.128
img[30,20,50];  # value of this voxel

image(img[,,50], col=gray(0:64/64), xlab="", ylab="", axes=FALSE, useRaster=TRUE);   # plot slice k=50
image(img[,20,], col=gray(0:64/64), xlab="", ylab="", axes=FALSE, useRaster=TRUE);   # plot slice j=20

# then similar for gifti? At least loading and getting parcel-average values out, maybe (show how vector, not 3d)


```