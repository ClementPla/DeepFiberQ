# DN-AI

This is the official repository for DN-AI, an automated tool for measurement of differentiated DNA replication in fluorescence microscopy images.

DN-AI offers different solutions for biologists to measure DNA replication in fluorescence microscopy images, without requiring programming skills. See the [Installation](#installation) section for instructions on how to install DN-AI.

## Features

- **Automated DNA replication measurement**: DN-AI can automatically measure the amount of DNA replication in fluorescence microscopy images. We use a deep learning model to segment the images and measure the amount of DNA replication.
- **User-friendly interface**: DN-AI provides a web-based user-friendly interface that allows users to easily upload images and view the results. Both jpeg and tiff images are supported.
- **Batch processing**: DN-AI can process multiple images at once, making it easy to analyze large datasets. It also supports comparing ratios between different batches of images.


## Installation

DN-AI relies on Python. We recommend installing its latest version (3.10 or higher) and using a virtual environment to avoid conflicts with other packages.

### Prerequisites
Before installing DN-AI, make sure you have the following prerequisites installed:
- [Python 3.10 or higher](https://www.python.org/downloads/) 
- [pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)

### Python Package
To install DN-AI as a Python package, you can use pip:

```bash
pip install git+https://github.com/ClementPla/DeepFiberQ.git
```


### Graphical User Interface (GUI)

To run the DN-AI graphical user interface, you can use the following command:

```bash
DNAI
```

Make sure you are running this command in the terminal where you have installed DN-AI. This will start a local web server and you will see output similar to:


Then open your web browser and go to `http://localhost:8501` to access the DN-AI interface.

Screenshots of the GUI:

![DN-AI GUI](imgs/screenshot.png)



### Docker
A Docker image is available for DN-AI. You can pull the image from Docker Hub:

```bash
docker pull clementpla/dnafiber
```

### Google Colab
We also provide a Google Colab notebook for DN-AI. You can access it [here](https://colab.research.google.com/github/ClementPla/DeepFiberQ/blob/main/Colab/DNA_Fiber_Q.ipynb).

