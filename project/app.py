from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np

app = Flask(__name__)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_python_code', methods=['POST'])
def run_python_code():
    uploaded_image = request.files['image']
    if uploaded_image:
        image_path = os.path.join("static/uploaded_images", uploaded_image.filename)
        uploaded_image.save(image_path)
    else:
        return jsonify({"error": "이미지를 업로드하세요."})

    result = process_image(image_path)

    return jsonify({"result": result})

@app.route('/show_image', methods=['POST'])
def show_image():
    image_filename = request.form['image_filename']
    if image_filename:
        image_path = os.path.join("static/output_images", image_filename)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
    return jsonify({"error": "해당 이미지를 찾을 수 없습니다."})

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_folder = "static/output_images"
    os.makedirs(output_folder, exist_ok=True) 
    output_image_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, img)
    return os.path.basename(output_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)