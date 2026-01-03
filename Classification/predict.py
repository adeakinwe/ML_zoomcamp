import pickle

saved_model = 'model_1.0.bin'

with open(saved_model, 'rb') as f_in:
    dv,model = pickle.load(f_in)


# PREDICTION FOR A NEW CUSTOMER

customer = {
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

def predict_customer_churn(customer):
    cust = dv.transform([customer])
    prediction = model.predict_proba(cust)

    return prediction[0,1]

prediction = predict_customer_churn(customer)
print(prediction)