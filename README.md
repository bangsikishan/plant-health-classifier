
# Image Classification Project

  

## Overview

This project implements an image classification service using a MobileNetV2 model to classify images of plants into three categories: angular leaf spot, bean rust, and healthy. The service is built using Flask and utilizes PyTorch for model training and inference. This application is particularly useful for agricultural applications, helping farmers and researchers identify plant diseases quickly and accurately.

  

## Table of Contents

- [Installation](#installation)

- [Usage](#usage)

- [API Endpoints](#api-endpoints)

- [Model Architecture](#model-architecture)

- [Dataset Information](#dataset-information)

- [Contributing](#contributing)

- [Future Work](#future-work)

- [License](#license)

- [Acknowledgments](#acknowledgments)

  

## Installation

  

### Prerequisites

- Python 3.7 or higher

- pip (Python package installer)

  

### Required Packages

The following packages are required to run the project:

- Flask==2.0.1

- torch==1.9.0

- torchvision==0.10.0

- Pillow==8.2.0

- tqdm==4.62.3

- transformers==4.11.3

- flask-cors==3.10.9

  

### Steps

1. Clone the repository:

```bash

git clone https://github.com/bangsikishan/plant-health-classifier.git

cd image_classification

```

  

2. Install the required packages:

```bash

pip install -r requirements.txt

```

  

3. Ensure you have the necessary model weights saved in the `image_classification_model/` directory.

  

## Usage

  

1. Start the Flask application:

```bash

python app.py

```

  

2. Open your web browser and navigate to `http://localhost:5000` to access the application.

  

3. To make predictions, send a POST request to the `/predict` endpoint with an image file:

```bash

curl -X POST -F "file=@path_to_your_image.jpg" http://localhost:5000/predict

```

  

4. The response will include the predicted class of the image.

  

## API Endpoints

  

-  **GET /**: Serves the `index.html` file, which provides a user interface for uploading images.

-  **POST /predict**: Accepts an image file and returns the predicted class in JSON format.

  

## Model Architecture

The model used in this project is MobileNetV2, a lightweight deep learning model designed for mobile and edge devices. It is efficient in terms of both speed and accuracy. The model is trained using knowledge distillation, where a smaller student model (MobileNetV2) learns from a larger teacher model.

  

## Dataset Information

The model is trained on a dataset of plant images, specifically focusing on three classes:

- Angular Leaf Spot

- Bean Rust

- Healthy

  

The dataset can be obtained from various agricultural datasets available online. Ensure that the images are labeled correctly for effective training.

  

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add new feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Create a new Pull Request.

  

## Future Work

- Expand the model to classify additional plant diseases.

- Implement a user authentication system for the web application.

- Enhance the front-end interface for better user experience.

- Explore other model architectures for improved accuracy.

  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

  

## Acknowledgments

- PyTorch for the deep learning framework.

- Flask for the web framework.

- HuggingFace for the teacher model.

- The contributors and maintainers of the datasets used.