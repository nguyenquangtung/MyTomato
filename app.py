import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from keras.preprocessing.image import img_to_array, load_img
from flask_cors import CORS

def preprocess_img(img):
    pre_img = load_img(img, target_size=(200, 200))
    img_array = img_to_array(pre_img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_class(prediction):
    # Assuming the prediction is an array of probabilities for each class
    predicted_class = np.argmax(prediction, axis=-1)
    Disease = ['Bac la muon', 'Bac la som', 'Khoe manh','Moc la', 'Nhen ve hai dom',
               'Vang xoan la','Virus kham la','Diem muc tieu','Dom la nau','Dom vi khuan']
    if np.all(0 <= predicted_class) and np.all(predicted_class <= 9):
        return Disease[predicted_class[0]]  # Use the first element if it's an array
    else:
        return "Unknown Disease"

# import request
app = Flask(__name__)
CORS(app)
model1 = tf.keras.models.load_model('batsgirls-leaves.h5')
model2 = tf.keras.models.load_model('Tomato_ripeness01.h5')

def predict_leaf_disease(image):
    # Load the model

    # Predict using the model
    prediction = model1.predict(image)
    # Get the predicted disease class label
    predicted_disease = predict_class(prediction)
    return predicted_disease

def predict_fruit_maturity(image):

    # Preprocess image if needed based on the model's input requirements
    prediction = model2.predict(image)
    maturity = np.argmax(prediction)
    return maturity


@app.route('/predict', methods=['GET']) # Nếu chạy trên localhost thì em dùng phương thức get để test, nếu backend gửi request thì dùng phương thức post để lấy hình từ backend gửi qua để predict
def predict():
    image = request.files['image'].read()
    # image = preprocess_img('test.jpg')
    disease = predict_leaf_disease(image)
    # maturity = predict_fruit_maturity(image)
    result = {
        'disease': disease,
        # 'maturity': maturity
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)