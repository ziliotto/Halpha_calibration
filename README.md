Python code for continuum subtraction and calibration of H-Alpha narrow-band imaging data. 
See details of the method in Chapter 3 of my master thesis: https://repositorio.uc.cl/xmlui/handle/11534/64798.

In summary, *heatmap.py* makes use of observations in different bands (g, r, i, which can be modified) to fit temperatures to each pixel of an image.
*calibrateline.py* simultaneously subtracts the continuum, with the shape of a Planck function, and calibrates the data. It is necessary to provide filter transmission functions, extinction coefficients and other parameters related with the observations for proper flux calibration.
