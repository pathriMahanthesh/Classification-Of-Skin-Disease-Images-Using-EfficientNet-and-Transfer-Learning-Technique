from flask import *
import os,sys
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/model_parameter')
def model_parameter():
    return render_template('model_parameter.html')

@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename))

@app.route('/submit_detector', methods=['POST'])
def choose_file():
    global file_path
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/uploads/' + f.filename)
        file_path = os.path.abspath('static/uploads/' + f.filename)

        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.vgg16 import preprocess_input
        from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
        # Set seeds for reproducibility
        np.random.seed(0)
        tf.random.set_seed(0)

        # Load the trained models
        model_cnn = load_model(r'1_cnn.h5')
        model_effnet = load_model(r'2_effnet.h5')
        model_densenet = load_model(r'3_densenet.h5')

        # Define class labels and descriptions for each model
        class_labels_cnn = {
            0: 'Actinic Keratosis',
            1: 'Basal Cell Carcinoma',
            2: 'Dermatofibroma',
            3: 'Melanoma',
            4: 'Nevus',
            5: 'Pigmented Benign Keratosis',
            6: 'Seborrheic Keratosis',
            7: 'Squamous Cell Carcinoma',
            8: 'Vascular Lesion'
        }

        class_descriptions_effnet = {
            0: 'pigmented benign keratosis',
            1: 'melanoma',
            2: 'vascular lesion',
            3: 'actinic keratosis',
            4: 'squamous cell carcinoma',
            5: 'basal cell carcinoma',
            6: 'seborrheic keratosis',
            7: 'dermatofibroma',
            8: 'nevus'
        }


        class_descriptions_densenet = {
            0: 'pigmented benign keratosis',
            1: 'melanoma',
            2: 'vascular lesion',
            3: 'actinic keratosis',
            4: 'squamous cell carcinoma',
            5: 'basal cell carcinoma',
            6: 'seborrheic keratosis',
            7: 'dermatofibroma',
            8: 'nevus'
        }

        # Define functions to preprocess images and make predictions for each model
        def classify_image_cnn(image_path):
            img = image.load_img(image_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            predictions = model_cnn.predict(img)
            class_index = np.argmax(predictions)
            predicted_class = class_labels_cnn[class_index]
            confidence = predictions[0][class_index]

            return predicted_class, confidence

        def classify_image_effnet(image_path):
            img = image.load_img(image_path, target_size=(75, 100))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input_densenet(img)

            predictions = model_effnet.predict(img)
            class_index = np.argmax(predictions)
            predicted_class = class_descriptions_effnet[class_index]
            confidence = predictions[0][class_index]

            return predicted_class, confidence


        def classify_image_densenet(image_path):
            img = image.load_img(image_path, target_size=(75, 100))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input_densenet(img)

            predictions = model_densenet.predict(img)
            class_index = np.argmax(predictions)
            predicted_class = class_descriptions_densenet[class_index]
            confidence = predictions[0][class_index]

            return predicted_class, confidence

        # Example usage:
        image_path = file_path # binod take this image input from webpage

        predicted_class_cnn, confidence_cnn = classify_image_cnn(image_path)
        predicted_class_effnet, confidence_effnet = classify_image_effnet(image_path)
        predicted_class_densenet, confidence_densenet = classify_image_densenet(image_path)

        print(f'CNN Predicted Class: {predicted_class_cnn}')  # display this result on webpage

        print(f'effnet Predicted Class: {predicted_class_effnet}')  # display this result on webpage also

        print(f'DenseNet Predicted Class: {predicted_class_densenet}')  # display this result on webpage also

        return render_template('detector.html',filename=f.filename, name=predicted_class_effnet)



if __name__=='__main__':
    app.run(debug=True)