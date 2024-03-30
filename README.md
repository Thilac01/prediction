# Linear Regression Model for Diabetes Prediction

This repository contains a simple implementation of a linear regression model to predict diabetes percentage based on age, blood pressure, and insulin levels. The model is trained using gradient descent optimization.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Installation

1. Clone this repository:

    ```bash
    git clone  https://github.com/Thilac01/prediction
    ```

2. Install the required Python packages:

    ```bash
    pip install pandas
    pip install joblib
    ```

## Usage

1. Prepare your dataset in CSV format. Ensure it contains columns for 'Age', 'BloodPressure', 'Insulin', and 'Outcome'.

2. Train the model by running `train_model.py`:

    ```bash
    python train_model.py
    ```

3. Once the model is trained, you can make predictions on new data by modifying the `age`, `blood_pressure`, and `insulin` variables in `predict.py` and running:

    ```bash
    python predict.py
    ```

## Model Persistence

The trained model parameters are saved using joblib in `linear_regression_model.pkl`. You can load this file to reuse the trained model without needing to retrain it.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)
