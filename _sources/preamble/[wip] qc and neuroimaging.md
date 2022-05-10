# QC and Neuroimaging

At the outset, QC failures can have widespread repercussions in the outcomes of neuroimaging analyses. 
For example, cascading errors in QC may introduce <ins>***variance and biases***</ins> into derived biomarkers/imaging phenotypes, which may lead to 
either an incorrect diagnosis of a condition or may obscure the presence of abnormalities in the tissue of interest; 
lack of correlation between the quantitative parameters detected and the qualitative appearance of the scan; etc. 
*Manual inspection* and *visual quality checks by expert raters* have been the conventional <ins>gold-standard</ins> for performing post-acquisition QC. 

However, manual quality checks are ***tedious and time-consuming*** especially in the presence of large and continuously evolving datasets ([Esteban et al. 2017](https://doi.org/10.1371/journal.pone.0184661); [Fidel Alfaro Almagro et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29079522/)) and may lack objectivity. 
Furthermore, with an *increase in dataset size and inspection times*, these checks are also affected by other *factors involving the raters* -such as training, experience, fatigue, bias and human-errors, motivation, among others- 
and *protocols* - such as clarity, levels of training, etc. Another variability can be induced by the *screen set-up* at the rater’s end - such as screen size, screen resolution and fidelity of the color representations. 
Also *image manipulation* such as adjusting contrast, brightness or color scales may induce variability and bias in the visual quality ratings.

## QC of acquired data

The QC of acquired data has received primary attention throughout the literature. 
On the one hand, acquisition is the entry point of the neuroimaging research workflow and therefore undetected defects in quality will cascade throughout the whole downstream pipeline. 
For instance, a failed prescription of the field of view partially clipping out a region of interest may successfully reach the analysis endpoint, snowballing errors as it progresses throughout the pipeline. 
On the other hand, the existence of physical phantoms and ex-vivo specimens that are widely used in the QA process of the hardware and foundational elements of the scanner’s software have been applied to bridge gaps between reference and no reference IQMs for QC. 
Finally, the cost of a quality issue generally is larger as it occurs earlier in the research pipeline. 
For instance, in the case of brain extraction, if one tool fails there might be an alternative software that procures sufficient quality. 
However, if the problem is that the acquisition is substandard, it is well possible that no tool will be able to perform sufficiently. 
This higher cost is also related to the difficulty of amending errors: taking our previous example of the prescription of the field of view, QA of such an error involves recalling the participant to run the MR protocol again. 
If the operator notices the problem during the session, recovering from this error will be less costly than realizing the problem after the participant has abandoned the facility. 
Even more deleterious are design issues and flaws of the standard operation procedure of the experiment, as they propagate across participants ([Bissett et al. 2021](https://doi.org/10.7554/eLife.60185)).

### Artifacts in the acquired data

MRI scans are susceptible to a wide-range of artifacts mainly due to interaction between the *object being imaged*, the *image acquisition sequence* and the *associated imaging hardware*. 
These generally include the following: 

*motion-induced artifacts*; 

*ringing artifacts driven by aliasing are also common*; 

*distortions due to gradient effects especially in accelerated image acquisition sequences*; 

*eddy current effects*; 

*magnetic field inhomogeneity (B0 and B1)* observed across different tissue regions; 

*coil-interference artifacts* commonly seen in parallel imaging techniques caused due to the use of multi-channel coils;

*zero filling artifacts*; 

*voxel-bleeding* resulting in overlapping of information; 

*chemical shift errors* due to the differences between resonance frequencies of fat and water;

and *aliasing artifacts* along with incomplete tissue details resulting from a mismatch between the field of view and the tissue region being scanned.

Among these, *motion artifacts* (head, eyes/blinking, swallowing, respiration, heartbeat, etc), *EM intereference* (spikes, zippers, etc), *thermal noise*, *temporal drifts in sensitivity*, *magnetic susceptibility artifacts* and the *presence of air cavities/bubbles* are found in both in-vivo and ex-vivo scans. 

The overall quality of the acquired scans can be affected by other factors such as: *species*, *age* (e.g. lower contrast between GM and WM in older patients compared to younger adults at 7T MRI which increases the chances of misclassification that can impact the segmentation quality), 
*gender, race, education, social environment, weight, height, medical conditions* (ex.: adhd tremors from Parkinson’s disease, etc.), 
*brain lesions and brain pathology* (e.g. may present as hyper- or hypointensities in MR images), *deviation from neurotypical development*, 
*standard operating procedures* (and fulfillment thereof) etc. of subjects.

From an image processing perspective, there would also be constraints in evaluating registration quality (when co-registering functional to anatomical scans), evaluation of segmentation quality (especially at the GM/WM boundaries) 
without having a well-defined decision boundary to assess the segmentations and effects of partial voluming.
