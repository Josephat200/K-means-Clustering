# Iris Classification

This project walks through a complete beginner-friendly machine learning workflow using the Iris dataset.

## Goal
Predict the species of an iris flower from four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target classes are:

- Setosa
- Versicolor
- Virginica

## Project Structure

- `dataset/` - place raw data here if you extend the project
- `notebooks/` - Jupyter notebooks for exploration and experiments
- `models/` - saved model artifacts
- `images/` - charts and visualizations
- `iris_classification.py` - end-to-end training script

## Workflow

1. Load the Iris dataset
2. Convert it to a pandas DataFrame
3. Explore the data with basic EDA
4. Visualize feature relationships
5. Split the data into train and test sets
6. Scale the features
7. Train a Logistic Regression model
8. Evaluate predictions
9. Save the trained model and scaler

## Install Dependencies

If you are using a virtual environment, install the packages from `requirements.txt`.

## Run the Script

```bash
python iris_classification.py
```

## Notebook

Open the notebook in `notebooks/iris_classification_basics.ipynb` for a step-by-step version of the workflow.
