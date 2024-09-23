# Brats Machine Learning Predictor - Web Application 🧠

The Brats ML Predictor Web Application is a powerful and user-friendly tool designed for the medical imaging community. It leverages state-of-the-art machine learning models to predict and analyze brain tumors using MRI scans. This application is built on top of robust frameworks, ensuring accuracy, reliability, and ease of use for healthcare professionals, researchers, and data scientists. This application, developed as part of the **Master in Computer Science and Information Systems** program at the [**Hellenic Open University (HOU)**](https://www.eap.gr/en/postgraduate-specialization-in-information-systems/), for my Diploma Thesis with title **'Implementation of a medical image classification system using deep learning techniques'** and implements the model trained in my repository [MSc Machine Learning Project](https://github.com/jimsnns/MSc_ML_Project/tree/master).

# Key Features

- Automated Tumor Detection: Utilizes advanced machine learning algorithms to detect and classify brain tumors from MRI scans with high precision.
- User-Friendly Interface: Intuitive and easy-to-navigate web interface, enabling users to upload MRI images and receive predictions effortlessly.
- Detailed Predictions: Provides comprehensive prediction reports, including tumor segmentation, tumor type classification, and probabilistic assessments.
- Visualization Tools: Integrated visualization tools to view MRI scans alongside the predicted tumor regions, facilitating better understanding and analysis.

# Technologies Used

- Streamlit: An open-source Python library for building interactive web applications, particularly in the realm of data science and machine learning   
- Machine Learning: Incorporates cutting-edge deep learning architectures trained on the BraTS dataset, ensuring high accuracy in tumor detection and classification.
- Data Storage: Utilizes efficient data storage solutions for managing MRI images and prediction results.

# Clone and Setup the environment locally

To get started with the Brats ML Predictor Web Application, follow these steps:

1. Clone the Repository:

        git clone https://github.com/jimsnns/brats_ml_predictor.git
        cd your-github-folder\brats_ml_predictor

3. Install Dependencies:

        pip install -r requirements.txt

4. Run the Application:

        python homepage.py

# Installation and Setup on Docker

To create and run a docker container and run the Brats ML Predictor Web Application, follow these steps:

1. Clone the Repository:

        git clone https://github.com/jimsnns/brats_ml_predictor.git
        cd brats_ml_predictor

2. Run the Application in Docker:

        docker build -t brats_ml_predictor .
        docker run -p 5000:5000 brats_ml_predictor
   
This setup will clone the repository, build the Docker image, and run the application in a Docker container, making it immediately ready for use.

  # Contributing

We welcome contributions from the community! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push the branch to your fork.
4. Submit a pull request with a detailed description of your changes.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.
