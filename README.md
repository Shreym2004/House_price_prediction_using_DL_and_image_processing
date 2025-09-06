# mumbai-house-price-prediction

Mumbai House Price Prediction — Project Overview

This project predicts residential house prices in Mumbai using a simple deep learning model implemented in MATLAB. The repository includes a self-contained MATLAB script that loads a CSV dataset, performs preprocessing (including fuzzy column matching, one-hot encoding for location, and normalization), trains a small feed-forward neural network, and allows interactive user predictions. It also visualizes the predicted property's approximate location on a map.

Key features:
- Robust column-name fuzzy matching to handle differing dataset schemas.
- One-hot encoding of locations and optional use of latitude/longitude if available.
- Normalization of inputs and target (price) for stable model training.
- A lightweight neural architecture (feature input → fully connected layers → regression output) trained with Adam optimizer.
- Interactive prompt for user inputs and map-based visualization using geoscatter.

How to use:
1. Place your dataset named 'house_prices_mumbai.csv' in this folder (or select it when prompted).
2. Ensure the dataset contains at least Location, Price, and one or more of: SquareFootage / Area, Bedrooms, Bathrooms. The script will attempt fuzzy matching for column names.
3. Open MATLAB, navigate to this folder, and run `predict_house_price.m`.
4. Follow prompts to input location, area, bedrooms, and bathrooms. The script will predict and display an estimated price (formatted in Lakhs/Crores) and show the location on a map.

Notes & next steps:
- This is a starter project designed for experimentation and improvement. For production use consider cross-validation, hyperparameter tuning, feature engineering (age, floor, amenities), outlier handling, and converting to Python/Scikit-learn or TensorFlow for easier deployment.
- The repository includes a sample `.gitignore`, LICENSE (MIT), and a requirements file listing compatible MATLAB versions/Toolboxes.

