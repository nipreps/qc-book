# Automating the quality assessment of MRI with MRIQC

## Synopsis
In this session, once a brief
introduction to the MRIQC
(Esteban et al., Plos One
2017) software is done, we
will use the image quality
metrics that the tool
generates to train and cross-validate a classifier,
implemented with the widely
used package scikit-learn.

Using interactive Jupyter
Notebooks that will be
available to the attendees
with sufficient time prior to
the meeting, we will explore
the features that we feed into
the classifier, underscoring
the so-called “batch-effects”
(as they are referred to in
molecular biology) due to the
provenance from different
acquisition devices
(“scanner-effects”, in the
case of MRI). We will dive
into some methodological
contents, investigating how
to best set up cross-validation when these
“scanner-effects” are
expected. For the training
and evaluation, we will use
openly available data already
processed with MRIQC. We
will also demonstrate a “real-world” application of interest
to researchers, which they
can run on their own data
(otherwise, new data will be
available to them). By
holding a dataset separated
from the cross-validation
framework, we will
demonstrate the process of
“calibrating” the classifier to
perform on their own
samples.
