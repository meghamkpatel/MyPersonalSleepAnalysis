import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import openai
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set API Keys and Paths
openai.api_key = "your_openai_api_key"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Google Sheets Setup
def connect_to_google_sheets(sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).sheet1

def save_to_google_sheets(sheet, dataframe):
    rows = dataframe.values.tolist()
    for row in rows:
        sheet.append_row(row)

# Sidebar Navigation
st.sidebar.title("Navigation")
views = ["Upload Data", "Daily Astrology", "Deep Dive + Chat"]
selected_view = st.sidebar.radio("Go to", views)

# **View 1: Upload Data**
if selected_view == "Upload Data":
    st.title("Upload Sleep Data")
    st.write("Upload sleep data images or CSV files for analysis and storage.")

    uploaded_file = st.file_uploader("Upload your Sleep Data (Image or CSV)", type=["png", "jpg", "jpeg", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            # Handle CSV
            uploaded_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(uploaded_data)

            # Save to Google Sheets
            sheet_name = "Sleep Data"  # Replace with your Google Sheet name
            sheet = connect_to_google_sheets(sheet_name)
            if st.button("Save to Google Sheets"):
                save_to_google_sheets(sheet, uploaded_data)
                st.success("Data saved to Google Sheets!")

        else:
            # Handle Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Extract text from the image
            text = pytesseract.image_to_string(image)
            st.write("Extracted Text:")
            st.text(text)

            # Save extracted text to Google Sheets
            if st.button("Save Image Data to Google Sheets"):
                sheet_name = "Sleep Data"  # Replace with your Google Sheet name
                sheet = connect_to_google_sheets(sheet_name)
                data_point = {"Raw Data": text}
                df = pd.DataFrame([data_point])
                save_to_google_sheets(sheet, df)
                st.success("Extracted data saved to Google Sheets!")

# **View 2: Daily Astrology**
elif selected_view == "Daily Astrology":
    st.title("Your Daily Astrology")
    st.write("Personalized insights for your day based on your data!")

    # Fetch data from Google Sheets
    sheet_name = "Sleep Data"  # Replace with your Google Sheet name
    sheet = connect_to_google_sheets(sheet_name)
    rows = sheet.get_all_records()
    if rows:
        central_data = pd.DataFrame(rows)

        # Generate personalized insights using OpenAI
        insights_prompt = f"""
        Using the following sleep data, generate a fun and personalized "daily astrology" insight:
        {central_data.tail(5).to_string(index=False)}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": insights_prompt}],
        )
        astrology_insight = response['choices'][0]['message']['content']
        st.write("✨ Your Daily Insight ✨")
        st.text(astrology_insight)
    else:
        st.warning("No data found. Upload your data in the 'Upload Data' section.")

# **View 3: Deep Dive + Chat**
elif selected_view == "Deep Dive + Chat":
    st.title("Deep Dive Analysis")
    st.write("Analyze your aggregated data and interact with the AI assistant.")

    # Fetch data from Google Sheets
    sheet_name = "Sleep Data"  # Replace with your Google Sheet name
    sheet = connect_to_google_sheets(sheet_name)
    rows = sheet.get_all_records()
    if rows:
        central_data = pd.DataFrame(rows)

        # Show aggregated data
        st.write("### Aggregated Data")
        st.dataframe(central_data)

        # Generate detailed analysis using OpenAI
        analysis_prompt = f"""
        Provide an in-depth analysis of the following sleep data. Include trends, suggestions, and areas for improvement:
        {central_data.to_string(index=False)}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
        )
        detailed_analysis = response['choices'][0]['message']['content']
        st.write("### AI Analysis")
        st.text(detailed_analysis)

        # Chat Feature
        st.write("### Chat with AI Assistant")
        user_input = st.text_input("Ask a question about your data, supplements, or workouts:")
        if user_input:
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert health assistant."},
                    {"role": "user", "content": user_input},
                ],
            )
            st.text(chat_response['choices'][0]['message']['content'])
    else:
        st.warning("No data found. Upload your data in the 'Upload Data' section.")
