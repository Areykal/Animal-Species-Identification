# Animal Species Identification Streamlit App

This is a Streamlit application for the Animal Species Identification project. Navigate to master branch for the code and implementation.

## Features
- Upload images for animal species identification
- Compare two-model approach vs end-to-end model
- View performance metrics and model information
- Interactive interface with visualizations

## Setup and Running
1. Install requirements:
pip install -r requirements.txt
2. Run the Streamlit app:
streamlit run app.py
3. Access the app in your web browser (typically at http://localhost:8501)

## Models
The application requires the following model files:
- `feature_extractor.h5`: The feature extraction model
- `svm_classifier.pkl` (or equivalent): The best classifier model
- `end_to_end_model.h5`: The end-to-end model
- `classifier_results.json`: Results from classifier training
- `model_comparison_results.json`: Comparison results between models

## Project Information
This application is part of the Animal Species Identification project for the AI class at American University of Phnom Penh.
