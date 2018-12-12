# Semantic Segmentation

This particular project utilizes semantic segmentation to apply an alpha mask in an area where hair is detected.  Simply create a clone of this directory, download the pre-trained weights (or import your own if you have used this architecture before) and run webcam.py.

## Dependencies
These programs were developed with Python 2.7.  Please have the following dependencies installed:
* keras (tensorflow)
* cv2
* numpy
* scipy
* skimage

## Instructions

1. Download the files Webcam.py and model.py.
2. Create a Python 2.7 virtual environment (optional)
3. Download pre-trained weights at: 
4. Run Webcam.py
	`python Webcam.py`
5. Allow OpenCV to use your webcam (if necessary).

## Future Work
A lot can still be done to improve the performance of this application (namely frame rate); any changes made will be uploaded to this repository.


## References & Resources

**Creating a python virtual environment:**
https://packaging.python.org/guides/installing-using-pip-and-virtualenv/

**Academic resources:**
Fischer, et al., U-Net: Convolutional Networks for Biomedical Image Segmentation. Retrieved From [https://arxiv.org/pdf/1505.04597.pdf](https://arxiv.org/pdf/1505.04597.pdf)  
Aarabi, Guo, et al., Real-time deep hair matting on mobile devices. Retrieved From [https://arxiv.org/pdf/1712.07168.pdf](https://arxiv.org/pdf/1712.07168.pdf)  
Balakrishna, et al., Automatic detection of lumen and media in the IVUS images using U-Net with VGG16 Encoder. Retrieved From [https://arxiv.org/pdf/1806.07554.pdf](https://arxiv.org/pdf/1806.07554.pdf)  
[https://github.com/MarcoForte/knn-matting](https://github.com/MarcoForte/knn-matting)