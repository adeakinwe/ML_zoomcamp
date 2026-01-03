import pickle

from flask import Flask
from flask import request
from flask import jsonify   

saved_model = 'model_1.0.bin'

with open(saved_model, 'rb') as f_in:
    dv,model = pickle.load(f_in)

app = Flask('churn-prediction')

@app.route('/predict', methods=['POST'])
def predict_customer_churn():
    customer = request.get_json()

    cust = dv.transform([customer])
    prediction = model.predict_proba(cust)[0,1]

    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8686)