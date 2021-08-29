# This is a repository for my semesterproject for EEG at university stuttgart

Feel free to visit: https://github.com/Jinaz/eeg_erp_tf_se  Here I uploaded my results. This zip should contain all needed things. If anything is missing feel free to drop me a message on ilias. Maybe some 3D framework is not doing its thing.

# N170  dataset
I chose this dataset because this one I could make the most out of. Imagining the experiment setup and procedure were the easiest for me on this one.
## running:

- install dependencies
- create "bids" folder and drop in the n170 dataset:  
root  
|_bids  
|&nbsp;&nbsp;&nbsp;|__n170  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___sub-001  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___sub-002  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___sub-003  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| ...  
|_img  
|_out_data  
|_python files

## dependencies
pandas, mne, mne_bids, numpy, matplotlib, autoreject (this one I only used for reference to compare results to), ccs_eeg_semesterproject, scipy  
Visulization:  
pysurfer, mayavi OR pyvista, tlk, pyvistaqt

## Content
- Cleaning of subjects 6,8, 13 with bad channels and segments in "out_data/"
- ERP Peak extraction for each subject for each condition
- statistical analysis of the extracted values
- Time frequency analysis per subject
- Time frequency analysis for a grand average
- Statistics for time frequency per subject 
- Source estimation per subject
- Source estimation for a grand average (in doing, needs so much time and ram doing this)
- Statistics for source estimation

## Report
- for each of the Contents above there is a notebook explaining decisions 
- There is a pdf report with: 
  - a short summary of the results
  - interpretation of the results
  - references to other research

## Result/Conclusion
With the full samplesize of 40 subjects covering all kinds of people, that recognize cars faster than faces and people that have difficulty recognize faces, one might try to generalize this on the general. As shown in the t-tests on the ERP peaks there is a significant difference recognizing faces and cars.   
This obviously is not the same for every person. On closer inspection there are subjects that have difficulties. I would recommend collecting more data and analyzing the outcome again to get a better sample size and more reliable results from it. The small sample size of 40 subjects can contain subjects that differ from the group and affect the results by a significant amount. 


