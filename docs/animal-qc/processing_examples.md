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
# Some configurations to "beautify" plots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Libre Franklin"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
```

# "What went wrong?" data processing edition
## Example 1
This example involves normalisation of a subject's anatomical image to a standard space template.
First, let's load the original image:
```{code-cell} python
from niwidgets import NiftiWidget

fname_orig = './assets/example-1_desc-orig_proc.nii.gz'
NiftiWidget(fname_orig)
```
and the template:
```{code-cell} python
tpl = './assets/example-1_desc-tpl_proc.nii.gz'
NiftiWidget(tpl)
```

Now, let's look at the processed image:
```{code-cell} python
fname_proc = './assets/example-1_desc-proc_proc.nii.gz'
NiftiWidget(fname_proc)
```

What do you think the problem was?
```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

```{admonition} Click the button to reveal the answer!
:class: dropdown
Wrong prior information: mouse target to rat template!

The reference image was a rat, but the target image was a mouse.
Although these species may have similar brain anatomy, the difference in scale was too large for the registration algorithm to overcome.

**To correct this**, make sure that the template and target image are as similar as possible.
```

## Example 2
This second example involves segmentation of a structural image into three tissue classes.
The resulting image can be viewed below:
```{code-cell} python
fname = './assets/example-2_proc.nii.gz'
NiftiWidget(fname)
```

What do you think the problem was?
```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

```{admonition} Click the button to reveal!
:class: dropdown
Wrong prior information: human tissue maps were used for prior probability!

**To correct this**, make sure that the prior information is species-specific.
```

## Example 3
This example used the default call to `N4BiasFieldCorrection` from `ANTs` to correct intensity non-uniformity across the image.
Again, let's look at the original first:

```{code-cell} python
fname = './assets/example-3_desc-orig_proc.nii.gz'
NiftiWidget(fname)
```

Now let's look at the corrected image. The results are not as good as they could be.

```{code-cell} python
fname = './assets/example-3_desc-n4_proc.nii.gz'
NiftiWidget(fname)
```

What do you think the problem was?
```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

````{admonition} Click the button to reveal!
:class: dropdown
Anisotropic voxel sizes with isotropic mm parameters

**To correct this**, define the number of elements in the bspline grid (including fewer elements in the slice direction), rather than using a bspline fitting distance which is given in mm.

```{code-cell} python
fname = './assets/example-3_desc-n4corr_proc.nii.gz'
NiftiWidget(fname)
```
````

## Example 4
This example concerns brain masking. First, we will use the popular FSL tool `bet` to remove non-brain tissue (aka skull-strip) an image.
This step can help downstream preprocessing such as registration.

```{code-cell} python
fname = './assets/example-4_desc-orig_proc.nii.gz'
NiftiWidget(fname)
```

After applying the mask supplied by `bet`, the image looks like this:
```{code-cell} python
bet_fname = './assets/example-4_desc-betmasked_proc.nii.gz'
NiftiWidget(bet_fname)
```

What do you think the problem was?
```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

```{admonition} Click the button to reveal!
:class: dropdown
`bet` assumes that the input data has a brain that is shaped like a humans (i.e., mostly spherical), but our data is more cylindrical.
```

Let's try another tool, instead this one is made for rats: `AFNI`'s `3dSkullStrip` with the `-rat` flag.
```{code-cell} python
3dss_fname = './assets/example-4_desc-3dssmasked_proc.nii.gz'
NiftiWidget(3dss_fname)
```

What do you think the problem was?
```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

````{admonition} Click the button to reveal!
:class: dropdown
Although the shape assumption is correct, `3dSkullStrip` assumes that the input image is T2-weighted. However, our image is T1-weighted.


**To correct this**, try using an atlas-based brain extraction method. Although this method takes longer than some of the other methods, it is robust to image contrast and across scanners. It is for this reason that the NiPreps tools, such as `NiRodents` and `NiBabies`, have developed their own versions of the `ANTs` atlas-based extraction tool: `antsBrainExtraction`.

```{code-cell} python
arts_fname = './assets/example-4_desc-artsmasked_proc.nii.gz'
NiftiWidget(arts_fname)
```
````

## Example 5
This example uses `FSL`'s `topup` tool to correct for susceptibility distortion in EPI images.
We have two spin-echo images, one with the phase-encoding direction that is superior to inferior, and the other with a phase-encoding direction going from inferior to superior.

Below are the mean images of the volumes acquired for each phase-encoding direction:
```{code-cell} python
fname_orig_SI = './assets/example-5_dir-SI_desc-orig_proc.nii.gz'
NiftiWidget(fname_orig_SI)
```

```{code-cell} python
fname_orig_IS = './assets/example-5_dir-IS_desc-orig_proc.nii.gz'
NiftiWidget(fname_orig_IS)
```

The `topup` output is less than desirable:
```{code-cell} python
fname_orig_IS = './assets/example-5_dir-IS_desc-orig_proc.nii.gz'
NiftiWidget(fname_orig_IS)
```

What do you think the problem was?
````{hint}
Check the image header:
```{code-cell} python
import nibabel as nb

img = nb.load(fname_orig_IS)
hdr = img.header()
print(hdr)
```
````

```{code-cell} python
:tags: [remove-input]
answer = input('What was the problem?')
print(answer)
```

```{admonition} Click the button to reveal!
:class: dropdown
Many `topup` parameters are defined in mm, which is too large for rodent voxel sizes.

**There's no perfect solution to correct for this (yet!)**. This is an example why image headers are manipulated to represent "human-sized" voxels. This approach is *not recommended*.

It may be possible to change the units of the header to microns and the voxel sizes accordingly. Although this still involves manipulation of the header, and the voxel sizes will be much larger than those of humans, it would probably have the same effect as scaling the header but while keeping the header information credible.

This approach still requires further testing.
```

