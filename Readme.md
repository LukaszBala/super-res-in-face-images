# Super Resolution in Face Images

This project focuses on performing super-resolution on face images using deep learning models. It enhances the resolution and quality of low-resolution face images, revealing finer details through a pretrained neural network model.

## Project Overview

The primary goal of this project is to upscale low-resolution face images while preserving key facial details and features. It uses the `uvicorn` library to run a Python server that provides an API for detecting faces in images and performing super-resolution on the detected faces.

## Requirements

- **Python 3.9**
- Required Python dependencies are listed in `requirements.txt`.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LukaszBala/super-res-in-face-images.git
cd super-res-in-face-images
```


### 2. Create a virtual environment (optional but recommended):

```bash
python3.9 -m venv venv
source venv/bin/activate
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Downloading Pretrained Weights

To utilize the pretrained models for super-resolution, download the pretrained weights into the `pretrained_models` directory.

- **Download Links**: Detailed download links are available in `pretrained_models/README.md`.

## Usage

To start the server, execute the following command:

```bash
python server.py
```

The server will run on port 8000 and provide the following API endpoints:

- **`/detect_faces`**: Detects faces in the provided images.
- **`/process_face`**: Performs super-resolution on the detected faces.


## Training

The training process utilizes the [RLFN](https://github.com/bytedance/RLFN) network.

To retrain the network, follow these steps:

1. **Run the Jupyter Notebook:**
   - Open and execute `master-thesis.ipynb`.

2. **Prepare Your Data:**
   - Create two directories: `train` and `valid`.
   - Inside each directory, place your images in separate subdirectories. For example:
     - `train/data/`
     - `valid/data/`

   Ensure that images are organized in their respective directories as shown above.

## License

The main project is licensed under the MIT License. See the `LICENSE` file for details.

This project incorporates code and models from several other projects. Specifically:

- **[GFPGAN](https://github.com/TencentARC/GFPGAN)** (Apache License 2.0)
- **[real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** (BSD 3-Clause License)
- **[RLFN](https://github.com/bytedance/RLFN)** (Apache License 2.0)

### Components and Modifications

- **Files in `gfpganModel/`**:
  - Taken and modified from the [GFPGAN project](https://github.com/TencentARC/GFPGAN).

- **Files in `realesrgan/`**:
  - Taken and modified from the [real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).

- **Files in `RLFN/`**:
  - Taken and modified from the [RLFN project](https://github.com/bytedance/RLFN).

Please refer to the respective project repositories for their licenses and additional information on the original code and models.
