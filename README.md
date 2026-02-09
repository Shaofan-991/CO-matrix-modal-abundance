\## Pixel-based modal abundance measurement from STEM-EDS maps



This repository contains the script used to quantify modal abundances of olivine and pyroxene from false-color STEM-EDS maps.



\### Method overview

The false-color EDS map is segmented using pixel classification in CIE-Lab color space.  

Pixels are assigned to olivine or pyroxene according to color similarity to representative color distributions derived from selected grains.  

Spatial consistency is enforced using connected-component majority voting, so that each mineral grain is classified as a single phase.



The procedure produces:

\- A classified phase map (green = olivine, blue = pyroxene)

\- Area fractions of both phases



\### Requirements

Python 3.9 or newer



Install required packages:



```

pip install -r requirements.txt

```



\### How to run

Place your EDS map (tiff or png) in the same folder as the script, then run:



```

python segmentation.py example_map.tiff

```



The output classified map will be saved automatically.



\### Example

An example EDS map (`example\_map.tiff`) is provided so the workflow can be reproduced.



\### Scientific purpose

This workflow was developed to measure modal abundances of submicron olivine and pyroxene in pristine CO chondrite matrices from STEM-EDS maps.  

The code is provided to ensure reproducibility of the quantitative petrographic measurements reported in the associated publication.


