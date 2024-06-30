import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

def load_model():
    model_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    model = hub.load(model_handle)
    return model

def detect_objects(image, model):
    resized_image = cv2.resize(image, (300, 300))
    input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def get_boxes_and_classes(detections, confidence_threshold=0.3):
    boxes, class_ids = [], []
    for i in range(int(detections['num_detections'].numpy()[0])):
        if detections['detection_scores'].numpy()[0][i] > confidence_threshold:
            box = detections['detection_boxes'].numpy()[0][i]
            class_id = int(detections['detection_classes'].numpy()[0][i])
            boxes.append(box)
            class_ids.append(class_id)
    return boxes, class_ids

def compare_boxes_and_classes(boxes1, boxes2, class_ids1, class_ids2, threshold=0.3):
    if len(boxes1) != len(boxes2) or set(class_ids1) != set(class_ids2):
        return False
    for box1, box2 in zip(boxes1, boxes2):
        if not np.allclose(box1, box2, atol=threshold):
            return False
    return True

def main():
    st.title("Image Similarity Detection")
    
    uploaded_image1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if uploaded_image1 and uploaded_image2:
        image1 = np.array(Image.open(uploaded_image1))
        image2 = np.array(Image.open(uploaded_image2))
        
        model = load_model()

        detections1 = detect_objects(image1, model)
        detections2 = detect_objects(image2, model)

        boxes1, class_ids1 = get_boxes_and_classes(detections1)
        boxes2, class_ids2 = get_boxes_and_classes(detections2)

        objects_in_same_position = compare_boxes_and_classes(boxes1, boxes2, class_ids1, class_ids2)

        if objects_in_same_position:
            st.image(image1, caption="First Image", use_column_width=True)
            st.image(image2, caption="Second Image", use_column_width=True)
            st.write("The images are similar.")
        else:
            st.image(image1, caption="First Image", use_column_width=True)
            st.image(image2, caption="Second Image", use_column_width=True)
            st.write("The images are not similar.")

if __name__ == "__main__":
    main()
