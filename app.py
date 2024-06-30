import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_dust(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply Canny edge detection to find edges
    edges = cv2.Canny(blurred, 30, 150)
    
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def draw_dust(image, contours, color=(0, 0, 255), thickness=2):
    # Draw contours on the image
    return cv2.drawContours(image.copy(), contours, -1, color, thickness)

def compare_dust_contours(contours1, contours2):
    # Compare the number of contours detected
    if len(contours1) != len(contours2):
        return False
    
    # Compare each contour (for simplicity, we compare the areas)
    for contour1, contour2 in zip(contours1, contours2):
        area1 = cv2.contourArea(contour1)
        area2 = cv2.contourArea(contour2)
        
        # Allow some difference in area due to noise
        if abs(area1 - area2) > 100:
            return False
    
    return True

def main():
    st.title("Dust Detection ")

    uploaded_image1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if uploaded_image1 and uploaded_image2:
        image1 = np.array(Image.open(uploaded_image1))
        image2 = np.array(Image.open(uploaded_image2))
        
        # Detect dust particles in both images
        contours1 = detect_dust(image1)
        contours2 = detect_dust(image2)

        # Draw contours on the images
        image1_with_dust = draw_dust(image1, contours1)
        image2_with_dust = draw_dust(image2, contours2)

        # Display images with detected dust
        st.image(image1_with_dust, caption="Image 1 with Detected Dust", use_column_width=True)
        st.image(image2_with_dust, caption="Image 2 with Detected Dust", use_column_width=True)

        # Check if dust particles are detected in both images
        dust_detected1 = len(contours1) > 0
        dust_detected2 = len(contours2) > 0

        if dust_detected1 and dust_detected2:
            # Compare dust patterns between the two images
            dust_similar = compare_dust_contours(contours1, contours2)
            if dust_similar:
                st.write("The images are similar.")
            else:
                st.write("The images are different.")
        else:
            st.write("At least one of the images has no dust particles detected.")

if __name__ == "__main__":
    main()
