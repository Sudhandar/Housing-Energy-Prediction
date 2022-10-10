import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():

#     int_features = [[int(x) for x in request.form.values()]]
#     features = pd.DataFrame(int_features, columns = ["regionc", "division",	"reportable_domain", "hdd65", "hdd30yr", "cdd30yr", "dollarel",	"dolelsph",	"metromicro", "ur",	"totrooms",	"heatroom",	"acrooms", "totsqft"])
#     # final_features = [np.array(int_features)]
#     prediction = model.predict(features)
#     output = json.dumps(prediction.item())
#     return render_template('index.html', prediction_text='Predicted Energy Consumption {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    features = pd.DataFrame([data], columns=data.keys())
    prediction = model.predict(features)
    output = json.dumps(prediction.item())
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)