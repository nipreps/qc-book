# Quality Control in preclinical MRI: Where do artifacts come from and how to fix them

## Synopsis
This talk will introduce
additional considerations for
quality assurance in small
animal MRI. For instance, we
have anecdotally found that
equipment for stabilising
physiology under anaesthetic
protocols can impact image
quality, particularly air
heaters. There are further
considerations to make after
the completion of each image
processing step, as many
software tools are designed
for humans and have implicit
voxel size/shape/contrast
assumptions that are not
valid for rodents.

The tutorial
will highlight where in the
processing pipeline these are
most likely to arise, examples
of these degraded results,
and suggestions to mitigate
them. Finally, we will frame
these problems in the context
of our experience extending
MRIQC and fMRIPrep for
rodents, to conclude with an
interactive session where the
participants will be
challenged with identifying
quality issues on a set of
images that conform a
“gallery of horrors,” which we
will build by gathering
examples showing artifacts
and quality issues from in-house databases, open
datasets, and examples
provided by the community.
The tutorial will demonstrate
the utility of the visual reports
MRIQC generates to identify
quality issues on preclinical
MRI data.

## What is preclinical imaging?

Preclinical imaging (i.e., imaging of experimental animal models) bridges the
gap between basic science and medical science by applying techniques from both
fields in the same individual.
For example, MRI can be combined with invasive procedures, such as optogenetic
techniques, in a single rodent subject but not in human participants.

## Why is preclinical imaging useful for conversations about quality?

In priniciple, preclinical MRI is equivalent to human MRI, which makes it very attractive as a translational technique.

However, in practice, preclinical MRI has historically trailed its human equivalent in some respects.
This likely reflects its novelty relative to human imaging, but there are a
number of factors including acquistion idiosyncrasies for a given study's design,
and the availability of software resources.

We will discuss how both of these factors can contribute to data quality,
regardless of the target species.
Following this, we will work through the stages of acquisitional and processing
standard operating procedures to visualise what happens when things go wrong and
how automated tools can objectively identify problematic images.


```{admonition} Learning outcomes
* Understand the similarities and differences between *preclinical* and human MRI
* Become familiar with the common sources of error in image acquisition and processing
* Understand the importance of visualisation for data quality 
* Use different visualisation methods to identify problems in image quality
```