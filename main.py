import pandas as pd
import joblib

model_params = joblib.load('linear_regression_model.pkl')
while True:

    age = int(input("Enter the Age: "))
    blood_pressure = int(input("Enter the Blood Presure: "))
    insulin = int(input("Enter the Insuline Leval: "))

    if age == 0:
        break

    prediction = (
        model_params['c1'] * age +
        model_params['c2'] * blood_pressure +
        model_params['c3'] * insulin
    )

    print("Predicted diabetes percentage:", prediction)
