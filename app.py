from flask import Flask, render_template, url_for, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image
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
        loaded_model = joblib.load(
            'Trained Model/breast-cancer/cancer_model.pkl')
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

# Diabetes


@app.route("/Diabet")
def diabet():
    return render_template("diabetes.html")


@app.route('/predictDiabet', methods=["POST"])
def predictDiabet():
    loaded_model = joblib.load('Trained Model/Diabet/diabetes_model.pkl')

    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)
        print(result)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))

# kidney disease


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


@app.route('/predictKD', methods=["POST"])
def predictKD():

    loaded_model = joblib.load('Trained Model/kidney/kidney_model.pkl')

    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))

# Liver disease


@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route('/predictLD', methods=["POST"])
def predictLD():

    loaded_model = joblib.load('Trained Model/liver/liver_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))

# brain tumor


@app.route('/BrainTumor')
def brain_tumor():
    return render_template('brain_tumor.html')


def predict_Btumor(img_path):
    model_load = load_model("Trained Model/brain tumor/brain_tumor.h5")

    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    preds = model_load.predict(img)
    return preds[0]


@app.route('/predictBTumor', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        preds = predict_Btumor(file_path)
        # print(preds)

        if int(preds[0]) == 0:
            result = "No worry! No Brain Tumor"
        else:
            result = "Patient has Brain Tumor"

        # print(f'prdicted: {result}')

        return result

    return None


# Stroke Disease
@app.route("/stroke")
def stroke():
    return render_template("stroke.html")


@app.route('/predictStroke', methods=["POST"])
def predictStroke():

    loaded_model = joblib.load('Trained Model/stroke/stroke_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


# Hepatitis
@app.route("/hepatitis")
def hepatitis():
    return render_template("hepatitis.html")


@app.route('/predictHepatitis', methods=["POST"])
def predictHepatitis():

    loaded_model = joblib.load('Trained Model/hepatitis/hepatitis_model.pkl')
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        result = loaded_model.predict(to_predict)

    if(int(result) == 1):
        prediction = "Sorry! it seems getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))


# ct scan images lung cancer detection
def predict_label(img_path):
	model = load_model("Trained Model/ct scan lung cancer/ct-scan-model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]

# routes
@app.route("/lung")
def lung():
	return render_template("lung.html")

@app.route("/preditLC", methods = ['GET', 'POST'])
def get_output():
	dic ={ 0:"Adenocarcinoma", 1:"Carcinoma", 2:"Normal", 3:"Squamous"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/ct scan images/" + img.filename
		img.save(img_path)

		p = np.argmax(predict_label(img_path))
		# print(p)

	return render_template("lung.html", prediction = dic[p], img_path = img_path)


if __name__ == "__main__":
    app.run(debug=True)
