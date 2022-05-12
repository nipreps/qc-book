# Idiosyncrasies of processing MRI data from small animals

## Rationale
Once the data are acquired, many of the same assumptions are true for human and
small animal imaging.
However, MRI processing frequently relies on prior information, and the default
assumption for many of the popular software tools is that the data are from healthy human adults.
This means that the most popular software tools are either adapted as needed to suit
a particular group, or lots of time and energy is used to create species-specific
emulations of these tools.

Instead, by including more flexibility at certain bottlenecks, image processing
software can be species agnostic, allowing all researchers to use the same tools
regardless of which species they were imaging.
These are many of the underlying changes made recently to MRIQC that faciliate its adaption to rats.

## Image scale
Due to the difference in brain size, preclinical imaging requires greater
resolution which is accounted for in part by the larger magnetic field strength
in preclinical scanners.
However, the scale of the resolution is not proportional to the voxel size, with
human voxel sizes often about ten times the size of rodent voxel sizes.

One hack that is used within preclinical imaging is to manipulate the image header
by multiplying the voxel images by ten to get "human" sized voxels.
This has serious implications for transparency and fidelity of the image header
and is not recommended.

Instead, software can take into account the voxel sizes when determining
distance-based parameters.
For example, a default gaussian kernel may have a full-width half-maximum (FWHM) of
6 mm, based on the assumption that the average voxel size is 2 mm isotropic.
Instead, the voxel size can be read from the image header, and define the FWHM as
three times the in-plane resolution.

## Image shape
In addition to the difference in voxel size, the shape of the images can also be very different.
For example, due to the elongation along the rostrocaudal plane relative to humans, the images often require more slices in the anterior-posterior direction to cover the whole brain.

In images from human data, the image origin may be manually set to the anterior commissure, which is near the centre of the image.
Although this landmark *can* be set as the origin in rodents, it is less likely to be located in the centre of the image.

## Prior information
Group-level analysis often requires the data to be transformed to a shared space.
Often, such a space is defined by a template image, which can be an average of many
images or a single reference image.
Templates are usually packaged with other resources, such as atlases, masks, and
probability maps, which delineate key regions of interest.

Most software packages ship their default resources, and some are "hard-coded" into
the algorithms so cannot be changed.
Thus, researchers working with small animals are more limited in their software
choices, or they must invest extra time into learning how to optimise the settings to account for a tangent from the defaults.

In MRIQC, prior information is provided by 
[TemplateFlow](https://www.templateflow.org), a cloud-based template repository.
Templates are programmatically selected as part of the workflow based on the
`--species` flag in the MRIQC command line call.

## Orientation and visualisation
Due to the traditional representation of rodent brain atlases, the coronal view is
generally the plane of choice for visualisating small animal data.
Additionally, the elongated rostrocaudal axis can be squashed when viewing the
sagittal plane if the visualisation software explains a less anisotropic field of
view.
