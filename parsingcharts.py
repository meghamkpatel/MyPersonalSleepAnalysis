import streamlit as st
from openai import OpenAI
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["general"]["openai_api_key"])

# Sleep Summary Extraction Function
def extract_sleep_data(image_path):
    """Extracts and organizes sleep data for analysis."""
    # Convert image to base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Prepare the prompt with base64 image content
    prompt = [
        {
            "type": "text",
            "text": "Provide a detailed analysis and insights based on the data, including:    1. Overall Sleep Summary (highlight patterns, quality, and anomalies).2. Efficiency and Quality (compare to recommended standards).    3. Blood Oxygen Trends (indicate concerns like dips or stability).    4. HRV Trends (highlight possible stress or recovery insights).    5. Heart Rate Trends (include any noteworthy patterns or changes).    6. Sleep Stage Trends and Comparison (identify balance among stages).    7. Correlations of Blood Oxygen, HRV, and Heart Rate over time. Provide actionable recommendations for improvement where applicable."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        }
    ]

    # Send request to OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()

# Visualization Function
def visualize_sleep_data(data):
    """Creates plots for sleep stage trends, HRV, and heart rate."""
    time = pd.to_datetime(data['time'])
    stages = data['stage'].map({"Awake": 3, "Light": 2, "Deep": 1, "REM": 0})
    heart_rate = data['heart_rate']

    # Sleep stages plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, stages, label="Sleep Stages", marker="o")
    plt.ylabel("Sleep Stages (REM=0 to Awake=3)")
    plt.xlabel("Time")
    plt.title("Sleep Stages Trend")
    plt.grid(True)
    plt.show()

    # HRV and heart rate comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time, heart_rate, label="Heart Rate", linestyle="--", color="red")
    plt.ylabel("Heart Rate (bpm)")
    plt.xlabel("Time")
    plt.title("Heart Rate and Sleep Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

# Sample Use Case
def main():
    # Placeholder path for the image
    image_path = r"C:\Users\Megha Patel\Downloads\Image-1 (1).jpg"   # Replace with actual path

    # Extract and analyze sleep data
    extracted_data = extract_sleep_data(image_path)
    print("Extracted Analysis:", extracted_data)

    # Visualization (if structured data is available)
    # visualize_sleep_data(parsed_data)

if __name__ == "__main__":
    main()
