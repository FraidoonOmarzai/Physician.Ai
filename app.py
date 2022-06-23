from flask import Flask, render_template, url_for, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html")


# Heart Prediction
@app.route("/heart")
def heart():
    return render_template("heart.html")


def PredictorHD(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if(size == 13):
        loaded_model = joblib.load("Trained Model/Heart-Model/heart_model.pkl")
        result = loaded_model.predict(to_predict)

    return result[0]


@app.route('/predictHD', methods=["POST"])
def predictHD():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        # print(to_predict_list)
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))
        # print(to_predict_list)
        if(len(to_predict_list) == 13):
            result = PredictorHD(to_predict_list, 13)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


# Breast Cancer prediction
@app.route("/BreastCancer")
def BreastCancer():
    return render_template("cancer.html")


def PredictorBC(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if(size == 13):
        loaded_model = joblib.load('Trained Model/breast-cancer/cancer_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictBC', methods=["POST"])
def predictBC():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        if(len(to_predict_list) == 13):
            result = PredictorBC(to_predict_list, 13)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


if __name__ == "__main__":
    app.run(debug=True)
