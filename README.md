# Machine Learning Screen Detection Project

This project involves the development of a machine learning system capable of detecting different types of screens—phone, tablet, and laptop—based on color sensor values (R, G, B, W). The screens are positioned at various distances (20cm, 30cm, 40cm, 50cm) to simulate real-world scenarios.

## Data Preprocessing
The raw sensor data undergoes preprocessing to enhance the quality and extract meaningful features. A median filter is applied to reduce noise, and feature extraction is performed by calculating the ratio of each color against the other colors. Additionally, temporal features are extracted by taking the ratio of past sensor values to present values.

## Machine Learning Models
Three machine learning models were employed in this study:
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

These models were chosen for their ability to handle the complexity and nuances of the sensor data.

## Evaluation
To assess the performance of the machine learning models, a classification report is generated for each. Furthermore, k-fold cross-validation is utilized to ensure the models' robustness and generalizability across different sets of data.

The combination of these techniques aims to create a reliable and accurate screen detection system using machine learning.
