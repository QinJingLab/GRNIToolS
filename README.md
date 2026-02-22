

# GRNIToolS: Gene Regulatory Network Tools Selector

**GRNIToolS** provides an efficient and reproducible evaluation pipeline for Gene Regulatory Network (GRN) inference. By integrating a wide array of inference methods into a suite of user-friendly tools, it accommodates different algorithm categories and platform requirements through three specialized modules.

**GitHub Repository:** [https://github.com/QinJingLab/GRNIToolS](https://github.com/QinJingLab/GRNIToolS)


## Modules Overview

The toolkit comprises one R package for classic statistical methods and evaluation, and two Python packages for deep learning and foundation model approaches.

For detailed installation, usage instructions, and examples, please refer to the specific `README.md` inside each module's respective directory.

### Module Architecture

<p align="center">
<img src="./img/image.png" alt="alt text" width="600"/>
</p>

---

### 1. GRNIToolS_R (Classic Methods)

* **Environment:** R v4.2.3
* **Description:** This core R package serves as the evaluation backbone and traditional inference engine. It wraps 31 GRN inference algorithms.
* **Features:** Integrates R-based algorithms using default parameters.
* **Documentation:** Please see the [`./GRNIToolS/README.md`] for full details.


### 2. GRNIToolS_DL (Deep Learning Module)

* **Environment:** Python 3.10, PyTorch 2.1.2
* **Description:** A dedicated Python module for deep learning-based GRN inference, including 10 algorithms.
* **Features:**
* Implements and trains all neural network architectures under a standardized protocol.
* Supports both default parameter settings and systematic hyperparameter optimization via the Optuna framework.
* **Documentation:** Please see the [`./GRNIToolS_DL/README.md`] for full details.



### 3. GRNIToolS_scFM (Single-Cell Foundation Models Module)

* **Environment:** Python 3.10, PyTorch 2.1.2
* **Description:** A Python module designed to extract gene embeddings from single-cell foundation models (scFMs) and use them as feature inputs for downstream GRN inference.
* **Features:**
* Integrates 9 scFMs for gene embedding extraction.
* Includes 9 downstream network prediction methods, spanning cosine similarity-based methods, classical machine learning models, and deep learning architectures.
* **Documentation:** Please see the [`./GRNIToolS_scFM/README.md`] for full details.

---

## Data and Resource Availability

To facilitate reproducibility and ease of use, large datasets, necessary model weights, and an offline version of our Docker image are hosted on Zenodo.

**Zenodo DOI:** (https://doi.org/10.5281/zenodo.18503007)

You can download the following resources from the Zenodo repository:

* **`GRNIToolS.zip`:** Contains the necessary datasets, pre-trained model checkpoints, and supplementary files required to run the full pipeline.
* **`grnitools_image.tar.gz`:** An offline archive of the complete Docker image. If you experience network issues pulling from Docker Hub, you can download this file and load the image locally using the command: `docker load -i grnitools_image.tar.gz`.


## Quick Start: Running via Docker

We provide an offline Docker image that contains the fully configured R and Python environments, complete with PyTorch and CUDA drivers, allowing you to run `GRNIToolS` directly without installing dependencies manually.

### 1. Load the Docker Image Locally

Make sure you have Docker installed. Download `grnitools_image.tar.gz` from our Zenodo repository and load it into your local Docker environment:

```bash
docker load -i grnitools_image.tar.gz
```

### 2. Run the Container
Start the container and mount your local data and script directories to the workspace inside the container.

```Bash
docker run -it --rm --gpus all grnitools:latest /bin/bash
```

## Contact


* **Yongqiang Zhou**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 

   **Email**: zhouyq67@mail2.sysu.edu.cn

* **Jing Qin**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 
   
   **Email**: qinj29@mail.sysu.edu.cn
---
