# Defining QC and QA

In many cases, QC and QA are known to be interlinked as **QA** is more *process-driven* to ensure a defect-free output/product,
while **QC** is *product-driven* that looks at the quality of the final output/product to see if it meets the final requirements.

## Quality Control (QC)

As per [Sreedher et al. (2021)](https://link.springer.com/article/10.1007/s00247-021-05043-6), **Quality Control (QC)** refers to “a real-time prospective process to ensure imaging quality is maintained 
by comparing it regularly to a defined set of criteria or industry standards.”
It is important to note that the ***product or output here*** could refer to either the scanner output or the output of some pre-processing or post-processing step.
For example, take the image acquisition step, QC refers then to a process of testing 
and finding possible quality issues in the output of image acquisition scanning protocols, 
and making sure that images not meeting pre-specified standards do not progress forward in the research workflow. 
QC is typically implemented by setting up quality checkpoints. 
This is done to ensure reliability and validity of brain imaging studies by ensuring the input images to each step 
in the processing workflow are within the domain of acceptable functional quality of such step. 
Therefore, QC is a ***forward-looking process*** (i.e., **prospective**).

Another caveat here is that these QC checkpoints also help define an "exclusion criteria" to discard certain units that do not meet the pre-defined standards. This exclusion criteria forms the crux of QC.

## Quality Assurance (QA)

**Quality assurance (QA)**, more broadly, is a way of preventing mistakes and defects in products and includes processes which are to be followed to ensure that the products meet a minimum standard. 
As per [Sreedher et al. (2021)](https://link.springer.com/article/10.1007/s00247-021-05043-6), “ISO 9000, a set of international standards developed by the International Organization for Standardization, defines QA as the part of quality management focused on providing confidence that quality requirements will be fulfilled”. 
In neuroimaging, QA focuses on improving image acquisition processes to mitigate any quality issues in the scanner output - 
these may involve aspects such as ensuring proper operation of the scanner, proper functioning of the gradient coils and the pre-installed magnet, ensuring subject stillness among others. 
Therefore, QA is a ***backward-looking process*** (i.e., **retrospective**) with the goal of preempting the replication of quality issues that eventually emerge in the output. 
Typically, QA utilizes the outputs of QC to identify these quality issues and then take corrective actions on the assessed system.

## Quality Improvement (QI)

Additionally, **Quality improvement (QI)** refers to all the ***data-driven processes with feedback*** which serve to improve imaging efficiency 
by identifying and quantifying critical metrics that define quality - these can also be seen to feed into the QA pipeline to bring the eventual
improvement in quality to meet the QC checkpoints.

## Image Quality Metrics 

***Image Quality Metrics (IQMs)*** are ***measurable aspects of the object under assessment*** that relate in some *direct or indirect way* to the underlying quality of the object. 
When a “perfect” or ideal instance of the object is available, IQMs are typically defined in reference to this ideal. 
Examples of referenced IQMs are common in the computer vision literature, for instance to compare and benchmark image compression algorithms, where high-quality, original images are subjected to different processes and then the outcomes of those processes are evaluated based on the fidelity of the output to the original image. 
However, and for most of the real-world cases, there is no access to the ideal instance of the object under assessment and therefore, “no-reference” IQMs must be defined.

An additional point to mention here is that quality can also depend on the scientific question one is trying to ask.  A subject's data may fail an IQM QC for one study because it has a bad segmentation in a certain region of the brain, 
but if in another study the focus is on another region, then it could pass your own IQM QC (while making the IQM QC of the former study not significant).
