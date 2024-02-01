
from flask import Flask
from flask import render_template, request
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json


print('Libraries are imported')



app = Flask(__name__)

# Load UNet model architecture and weights
with open('model\model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    
print('Architecture is imported')

model.load_weights('model\model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('weight is imported')


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((208, 208))  # Adjust the size based on your model's input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

print('image is preprocessed')

# Function to perform segmentation using the loaded model
def segment_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    segmented_img = model.predict(preprocessed_img)
    return segmented_img.squeeze()  # Remove the batch dimension

print('image is segmented')

os.makedirs('uploads', exist_ok=True)
os.makedirs(os.path.join('static', 'segmented_images'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment/', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Perform segmentation
        segmented_image = segment_image(image_path)
        print("Segmented Image Shape:", segmented_img.shape)
        # Save the segmented image
        segmented_image_path = os.path.join('static', 'segmented_images', file.filename)
        Image.fromarray((segmented_image * 255).astype(np.uint8)).convert('RGB').save(segmented_image_path)

        return render_template('index.html', segmented_image_url=segmented_image_path)
    

if __name__ == '__main__':
    app.run(debug=True)
    
#@app.route('/')
#def index():
   # return render_template('index.html')
#@app.route('/')
#def index_view():
#    return render_template('index.html')

#print("Index is called")

#def convertImage(imgData1):
    #imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    #with open('output.png', 'wb') as output:
    # output.write(base64.b64decode(imgstr))
    # print("Image is converted from base64")
    # @app.route('/segment/', methods=['POST'])
    # def segment():
    # if 'image' not in request.files:
    # return jsonify({'error': 'No file part'})
    # image = request.files['image']
    # if image.filename == '':
    # return jsonify({'error': 'No selected file'})
    # if image:
        # Perform image segmentation and save the segmented image to a file
        # segmented_image = perform_segmentation(image)
        # segmented_image.save('static/segmented_image.png')
        # segmented_image_url = url_for('static', filename='segmented_image.png')
        # response = {'segmented_image_url': segmented_image_url}
        # return jsonify(response)
        # print('predicted')
        # def perform_segmentation(image):
    # Preprocess the input image if needed (e.g., resizing)
    # ...

    # Assuming your model expects input in the shape (height, width, channels)
    # input_data = preprocess_input(image)

    # Perform segmentation using the loaded model
    # segmentation_mask = loaded_model.predict(np.expand_dims(input_data, axis=0))

    # Convert the segmentation mask to an image
    # segmented_array = np.argmax(segmentation_mask, axis=-1)
    # segmented_image = Image.fromarray((segmented_array * 255).astype(np.uint8))
    # return segmented_image
    # if __name__ == '__main__':
    # app.run(debug=True)

