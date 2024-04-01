from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')

model = joblib.load(open("student.pkl", "rb"))
studentInfo_df = pd.read_csv('studentInfo.csv')


# Route to render the homepage
@app.route('/')
def home():
    code_module_distribution = get_code_module_distribution()
    return render_template('upload.html', code_module_distribution=code_module_distribution)

# Route to handle form submission and make prediction
@app.route("/", methods=["POST"])
def predict():
    if request.method == "POST":
        '''float_features = [float(x) for x in request.form.values()]
        features = np.array(float_features[5:10]).reshape(1, -1)
        prediction = model.predict(features)
        classes = ["Pass", "Fail"]
        return render_template('predict.html', prediction_text_=classes[prediction[0]], u_id=float_features[2], u_m=float_features[1], cm=float_features[0])'''
        float_features = [x for x in request.form.values()]
        code_module = int(float_features[0])
        print("code_module", code_module)

        module_labels = {
            0: "AAA",
            1: "BBB",
            2: "CCC",
            3: "DDD",
            4: "EEE",
            5: "FFF"
        }

        # Using the dictionary to get the label for code_module
        code_module_label = module_labels.get(code_module)

        user_id = float_features[2]
        user_module = float_features[1]

        features = np.array(
            [float(x) for x in float_features[5:10]]).reshape(1, -1)

        print(features)
        prediction = model.predict(features)
        classes = ["pass", "Fail"]

        return render_template('predict.html', prediction_text_=classes[prediction[0]], u_id=user_id, u_m=user_module,
                               cm=code_module_label)

# Route to get code module distribution
@app.route('/get_code_module_distribution', methods=["GET"])
def get_code_module_distribution():
    code_module_distribution = studentInfo_df['code_module'].value_counts().to_dict()
    return jsonify(code_module_distribution)

# Other routes to get gender, age band, and region distribution
@app.route('/get_gender_distribution/<code_module>', methods=["GET"])
def get_gender_distribution(code_module):
    filtered_df = studentInfo_df[studentInfo_df['code_module'] == code_module]
    gender_distribution = filtered_df['gender'].value_counts().to_dict()
    return jsonify(gender_distribution)

@app.route('/get_age_band_distribution/<code_module>', methods=["GET"])
def get_age_band_distribution(code_module):
    filtered_df = studentInfo_df[studentInfo_df['code_module'] == code_module]
    age_band_distribution = filtered_df['age_band'].value_counts().to_dict()
    return jsonify(age_band_distribution)

@app.route('/get_region_distribution/<code_module>', methods=["GET"])
def get_region_distribution(code_module):
    filtered_df = studentInfo_df[studentInfo_df['code_module'] == code_module]
    region_distribution = filtered_df['region'].value_counts().to_dict()
    return jsonify(region_distribution)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,debug=True)