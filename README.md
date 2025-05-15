# Image Colorizer Web App

## Overview

This web application, built using Streamlit, allows you to colorize black and white images using deep learning models. It provides a user-friendly interface to upload an image and see it transformed into color. The app uses pre-trained models (SIGGRAPH 17 and ECCV 16) to perform the colorization. 

https://github.com/user-attachments/assets/c8cd9e03-eac8-4527-b93e-b64a01b2f477


## Features

* **Image Colorization:** Colorize black and white images using deep learning.
* **Model Selection:** Choose between two colorization models:
    * **ECCV 16:** Faster processing, less detailed results.
    * **SIGGRAPH 17:** Higher quality, more processing time.
    * **Both Models**: See the results from both models side-by-side.
* **Image Upload:** Upload images directly through the web interface (supports jpg, jpeg, png, and bmp formats).
* **Output Image Sizing:** Adjust the size of the output colorized image.
* **Download Results:** Download the colorized images.
* **User-Friendly Interface:** Simple and intuitive design with clear instructions.

## How It Works

1.  **Image Upload:** The user uploads a black and white image through the Streamlit interface.
2.  **Image Processing:**
    * The uploaded image is loaded and resized.
    * The image is converted to the Lab color space, where L represents lightness, and a and b represent color.
    * The L channel is separated and used as input to the colorization model.
3.  **Colorization:**
    * The selected deep learning model (ECCV 16 or SIGGRAPH 17) predicts the a and b channels.
    * The predicted a and b channels are combined with the original L channel.
4.  **Output:**
    * The combined Lab image is converted back to the RGB color space.
    * The original grayscale image and the colorized image are displayed.
    * The user can download the colorized image.

## Code Structure

The project contains the following files:

* **`README.md`:** This file, providing an overview of the project.
* **`app.py`:** The main Streamlit application code.  It handles the web interface, image processing, and model loading.
* **`requirements.txt`:** A list of Python packages required to run the application.
* **`.devcontainer/devcontainer.json`:** A configuration file for setting up a development environment using Docker.

### `app.py`

* **Imports:** Imports necessary libraries, including Streamlit, PIL, scikit-image, torch, and matplotlib.
* **Streamlit Setup:** Configures the Streamlit page layout and title.
* **Header and Description:** Displays the title and description of the app.
* **BaseColor Class:** Defines a base class for colorization models, including methods for normalizing and unnormalizing L and ab values.
* **ECCVGenerator Class:** Implements the ECCV 16 colorization model architecture.
* **SIGGRAPHGenerator Class:** Implements the SIGGRAPH 17 colorization model architecture.
* **Model Loading:** Loads the pre-trained ECCV 16 and SIGGRAPH 17 models.
* **Helper Functions:**
    * `load_img()`: Loads an image using PIL.
    * `resize_img()`: Resizes an image using PIL.
    * `preprocess_img()`: Prepares the image for the colorization model by resizing it and converting it to the Lab color space.
    * `postprocess_tens()`:  Post-processes the model output, resizing it to the original image size and converting it back to RGB.
    * `get_image_download_link()`: Generates an HTML link to download the colorized image.
* **Main Application Flow:**
    * Sets up the sidebar for model selection and image size.
    * Handles image uploads.
    * Loads the selected pre-trained model.
    * Processes the uploaded image.
    * Displays the original, grayscale, and colorized images.
    * Provides a download link for the colorized image.
* **Footer:** Displays a message indicating the use of deep learning colorization models.

### `requirements.txt`

Lists the Python packages required to run the application:

* `torch`: PyTorch for deep learning.
* `scikit-image`: For image processing.
* `numpy`: For numerical operations.
* `matplotlib`: For plotting (though not directly used for display in the final app, it might be used in the model definitions).
* `pillow`:  Python Imaging Library (PIL) for image manipulation.
* `streamlit`: For creating the web application.

### `.devcontainer/devcontainer.json`

This file is used to configure a development container using Docker. It specifies:

* The Docker image to use (`mcr.microsoft.com/devcontainers/python:3.11-bullseye`).
* VS Code settings and extensions.
* Commands to install dependencies (`packages.txt` and `requirements.txt`).
* A command to run the Streamlit app when the container is attached.
* Port forwarding configuration for the Streamlit app (port 8501).

## Setup (Using Docker - Recommended)

1.  **Install Docker:** Install Docker Desktop on your system.
2.  **Clone the Repository:** Clone the project repository to your local machine.
3.  **Open in VS Code (with Remote - Containers extension):**
    * Open the cloned repository in VS Code.
    * Install the "Remote - Containers" extension in VS Code.
    * Reopen the project in a container (VS Code will prompt you, or you can use the command palette).

    VS Code will automatically set up the development environment with all the necessary dependencies. The application will be accessible at `http://localhost:8501`.

## Setup (Without Docker - Local)

1.  **Install Python:** Ensure you have Python 3.11 or later installed.
2.  **Clone the Repository:** Clone the project repository to your local machine.
3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Application:**
    ```bash
    streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
    ```
    The application will be accessible at `http://localhost:8501`.

## Usage

1.  **Open the App:** Open the application in your web browser (usually at `http://localhost:8501`).
2.  **Upload Image:** Click on "Choose an image..." and select a black and white image from your computer.
3.  **Select Model (Optional):** Choose a colorization model from the sidebar:
    * "ECCV 16" for faster colorization.
    * "SIGGRAPH 17" for higher quality colorization.
    * "Both Models" to see both results.
4.  **Adjust Image Size (Optional):** Use the "Output Image Size" slider in the sidebar to change the resolution of the output image.
5.  **View Results:** The colorized image (or images) will be displayed below the upload section.
6.  **Download Image:** Click the "Download" link below the colorized image to save it.

## Disclaimer

The colorized images are generated by a deep learning model and may not always be perfect. The quality of the results can vary depending on the input image and the selected model.
