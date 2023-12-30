import streamlit as st
import requests
import cv2
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Streamlit layout
st.title("Stress Detection App")

# User input for step count, temperature, and humidity
step_count = st.number_input("Enter Step Count:")
temperature = st.number_input("Enter Temperature:")
humidity = st.number_input("Enter Humidity:")

# Button to get stress prediction
if st.button("Get Stress Prediction"):
    # Make an API request to get stress prediction with user input
    api_url = 'http://localhost:5000/process_frame'
    payload = {'step_count': step_count, 'temperature': temperature, 'humidity': humidity}
    response = requests.post(api_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        predicted_stress = response.json()['predicted_stress']
        st.success(f"Predicted Stress Level: {predicted_stress}")
    else:
        st.error("Failed to get stress prediction. Please check your input and try again.")

# Button to start video streaming
if st.button("Start Video Streaming"):
    # Make an API request to start video streaming
    api_url = 'http://localhost:5000/start_streaming'  # Update the URL
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Display streaming video
        st.video(response.content)
    else:
        st.error("Failed to start video streaming. Please check your input and try again.")

# Placeholder for the stress curves
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Run the real-time processing loop
while True:
    # Make an API request to get stress data
    api_url = 'http://localhost:5000/get_stress_data' 
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        stress_data = response.json()

        # Update plots for each factor
        axes[0, 0].clear()
        axes[0, 0].plot(stress_data["blink_stress"])
        axes[0, 0].set_title('Blink Stress')

        # Add other plots for different stress factors

        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)
        st.text("Note: Faces are not displayed in the streaming video.")
    else:
        st.error("Failed to get stress data. Please try again.")

    # Break the loop if 'Stop Streaming' button is pressed
    if st.button:
        break
