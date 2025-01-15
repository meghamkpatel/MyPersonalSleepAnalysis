import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd
import openai
import gspread
from google.oauth2.service_account import Credentials
import json
import re

# Set API Keys and Paths
openai_api_key = st.secrets["general"]["openai_api_key"]
gcs_credentials = {
    "type": "service_account",
    "project_id": st.secrets["gcs"]["project_id"],
    "private_key_id": st.secrets["gcs"]["private_key_id"],
    "private_key": st.secrets["gcs"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["gcs"]["client_email"],
    "client_id": st.secrets["gcs"]["client_id"],
    "auth_uri": st.secrets["gcs"]["auth_uri"],
    "token_uri": st.secrets["gcs"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcs"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcs"]["client_x509_cert_url"]
}
openai_client = openai.OpenAI(api_key=openai_api_key)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Google Sheets Setup
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS = Credentials.from_service_account_info(gcs_credentials, scopes=SCOPE)
gc = gspread.authorize(CREDS)

def connect_to_google_sheets(sheet_name):
    try:
        return gc.open(sheet_name).sheet1
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

def save_to_google_sheets(sheet, dataframe):
    try:
        # Get existing headers from the sheet
        existing_headers = sheet.row_values(1)  # Assumes headers are in the first row
        required_headers = list(dataframe.columns)

        # Add headers if the sheet is empty
        if not existing_headers:
            sheet.append_row(required_headers)
            existing_headers = required_headers

        # Ensure headers match
        if existing_headers != required_headers:
            st.error("The columns in the Google Sheet do not match the data being uploaded.")
            return

        # Align dataframe to match headers
        aligned_data = dataframe.reindex(columns=existing_headers, fill_value="")

        # Append rows in bulk
        rows = aligned_data.values.tolist()
        sheet.append_rows(rows)  # Bulk append for better performance
        
        st.success("Data saved to Google Sheets!")
    except Exception as e:
        st.error(f"Failed to save data to Google Sheets: {e}")

def clean_duration_field(duration):
    """Clean and validate Total Duration."""
    match = re.search(r"(\d+H\d+M)", duration)
    return match.group(1) if match else "Invalid Format"

def extract_structured_data_from_image(image):
    """Extract structured data using OCR with bounding box analysis."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    parsed_data = {
        "Date": "",
        "Total Duration": "",
        "Sleep Score": "",
        "Sleep Efficiency": "",
        "Sleep Quality": "",
        "Awake Time": "",
        "REM Duration": "",
        "Light Sleep Duration": "",
        "Deep Sleep Duration": "",
        "Average Heart Rate": "",
        "Average Blood Oxygen": "",
        "Average HRV": "",
        "Goal Achieved": "",
    }

    for i, text in enumerate(data['text']):
        text = text.strip()
        if "Total Duration" in text:
            duration_line = data['text'][i + 1:i + 3]
            valid_values = [x for x in duration_line if x.isdigit() or "H" in x or "M" in x]
            if valid_values:
                parsed_data["Total Duration"] = " ".join(valid_values[:-1])  # Remove sleep score
                parsed_data["Sleep Score"] = valid_values[-1]  # Sleep score is the last value
        elif "Sleep Efficiency" in text:
            parsed_data["Sleep Efficiency"] = text.split()[-1]
        elif "Sleep Quality" in text:
            parsed_data["Sleep Quality"] = text.split()[-1]
        elif "Total awake time" in text:
            parsed_data["Awake Time"] = text.split()[-1]
        elif "REM duration" in text:
            parsed_data["REM Duration"] = text.split()[-1]
        elif "Total light sleep duration" in text:
            parsed_data["Light Sleep Duration"] = text.split()[-1]
        elif "Total deep sleep duration" in text:
            parsed_data["Deep Sleep Duration"] = text.split()[-1]
        elif "Average heart rate" in text:
            parsed_data["Average Heart Rate"] = " ".join(text.split()[-2:])  # Account for "bpm"
        elif "Average blood oxygen" in text:
            parsed_data["Average Blood Oxygen"] = text.split()[-1]
        elif "Average HRV" in text:
            parsed_data["Average HRV"] = text.split()[-1]
        elif "Goal achieved" in text:
            parsed_data["Goal Achieved"] = " ".join(text.split()[-3:])

    parsed_data["Total Duration"] = clean_duration_field(parsed_data["Total Duration"])
    return pd.DataFrame([parsed_data])

def use_gpt_for_parsing(text):
    """Use GPT to parse and structure extracted OCR text."""
    prompt = f"""
        Parse the following sleep data into valid JSON with the following keys:
        Date, Total Duration, Sleep Score, Sleep Efficiency, Sleep Quality, Awake Time, REM Duration,
        Light Sleep Duration, Deep Sleep Duration, Average Heart Rate, Average Blood Oxygen,
        Average HRV, Goal Achieved.

        For example:
        Date	Total Duration	Sleep Score	Sleep Efficiency	Sleep Quality	Awake Time	REM Duration	Light Sleep Duration	Deep Sleep Duration	Average Heart Rate	Average Blood Oxygen	Average HRV	Goal Achieved
        Tue, Jan 14, 2025	08H11M	96	100%	Excellent	01H26M	16M	05H23M	01H38M	62 bpm	98%	39 ms	0 of 7 days

        {text}

        ONLY return a valid JSON object. Do not include explanations, examples, or any additional text.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        structured_data = response.choices[0].message.content.strip()

        # Validate and parse JSON
        try:
            parsed_data = json.loads(structured_data)
        except json.JSONDecodeError:
            # Extract JSON manually if response includes extra content
            start_idx = structured_data.find("{")
            end_idx = structured_data.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                structured_data = structured_data[start_idx:end_idx]
                parsed_data = json.loads(structured_data)
            else:
                st.error("Failed to parse GPT response as JSON. Ensure the response format is correct.")
                return None

        return pd.DataFrame([parsed_data]) if parsed_data else None

    except Exception as e:
        st.error(f"Error during GPT parsing: {e}")
        return None

def generate_sleep_summary(data_row):
    """
    Generate a sleep summary for the day based on the submitted sleep data.
    """
    # Extract relevant fields
    date = data_row.get("Date", "N/A")
    total_duration = data_row.get("Total Duration", "N/A")
    sleep_score = data_row.get("Sleep Score", "N/A")
    sleep_efficiency = data_row.get("Sleep Efficiency", "N/A")
    sleep_quality = data_row.get("Sleep Quality", "N/A")
    rem_duration = data_row.get("REM Duration", "N/A")
    deep_sleep = data_row.get("Deep Sleep Duration", "N/A")
    avg_hr = data_row.get("Average Heart Rate", "N/A")
    avg_spo2 = data_row.get("Average Blood Oxygen", "N/A")
    avg_hrv = data_row.get("Average HRV", "N/A")

    # GPT prompt for generating the summary
    prompt = f"""
    Based on the following sleep data:
    - Date: {date}
    - Total Duration: {total_duration}
    - Sleep Score: {sleep_score}
    - Sleep Efficiency: {sleep_efficiency}
    - Sleep Quality: {sleep_quality}
    - REM Duration: {rem_duration}
    - Deep Sleep Duration: {deep_sleep}
    - Average Heart Rate: {avg_hr}
    - Average Blood Oxygen: {avg_spo2}
    - Average HRV: {avg_hrv}

    Write a friendly and insightful summary about the user's sleep. Include any patterns, potential improvements, or highlights, such as excellent REM sleep or low blood oxygen levels.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating sleep summary: {e}")
        return None

def analyze_sleep_cycles(sleep_data):
    """
    Analyze sleep cycle data, visualize it, and provide insights.
    Parameters:
    - sleep_data (pd.DataFrame): Dataframe containing sleep stages, time, and heart rate.
    """
    # Check for required columns
    required_columns = ["time", "stage", "heart_rate"]
    for column in required_columns:
        if column not in sleep_data.columns:
            raise ValueError(f"Missing required column: {column}")

    # Sort data by time
    sleep_data["time"] = pd.to_datetime(sleep_data["time"], format="%H:%M")
    sleep_data = sleep_data.sort_values(by="time")

    # Plot sleep stages and heart rate
    fig, ax1 = plt.subplots(figsize=(12, 6))
    stages = sleep_data["stage"].map({"Awake": 3, "Light": 2, "Deep": 1, "REM": 0})
    ax1.plot(sleep_data["time"], stages, label="Sleep Stages", color="blue", marker="o")
    ax1.set_ylabel("Sleep Stages (0=REM, 1=Deep, 2=Light, 3=Awake)", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["REM", "Deep", "Light", "Awake"])
    ax1.tick_params(axis="y", colors="blue")
    ax1.grid(axis="x")

    ax2 = ax1.twinx()
    ax2.plot(sleep_data["time"], sleep_data["heart_rate"], label="Heart Rate", color="red", linestyle="--")
    ax2.set_ylabel("Heart Rate (bpm)", color="red")
    ax2.tick_params(axis="y", colors="red")

    fig.suptitle("Sleep Stages and Heart Rate", fontsize=16)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Generate insights
    longest_stage = sleep_data["stage"].value_counts().idxmax()
    lowest_hr_stage = sleep_data.loc[sleep_data["heart_rate"].idxmin(), "stage"]
    insights = f"""
    Sleep Analysis Insights:
    - Longest sleep stage: {longest_stage}
    - Stage with lowest heart rate: {lowest_hr_stage}
    - Average heart rate: {sleep_data['heart_rate'].mean():.2f} bpm
    """
    return insights

# New: Extract structured sleep data from image
def extract_structured_data_from_image(image):
    """Extract structured sleep data from OCR."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    times, stages, heart_rates = [], [], []

    for i, text in enumerate(data['text']):
        if text in ["Awake", "Light", "Deep", "REM"]:
            times.append(data['text'][i - 1])  # Assume time is the previous word
            stages.append(text)
            hr_candidates = data['text'][i + 1:i + 3]
            hr_value = next((val for val in hr_candidates if val.isdigit()), None)
            heart_rates.append(int(hr_value) if hr_value else None)

    # Create a DataFrame
    sleep_data = pd.DataFrame({"time": times, "stage": stages, "heart_rate": heart_rates})
    sleep_data["time"] = pd.to_datetime(sleep_data["time"], format="%H:%M", errors="coerce")
    return sleep_data.dropna().reset_index(drop=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
views = ["Upload Data", "Daily Astrology", "Deep Dive + Chat"]
selected_view = st.sidebar.radio("Go to", views)

# **View 1: Upload Data**
if selected_view == "Upload Data":
    st.title("Upload Sleep Data")
    st.write("Upload sleep data images for analysis and storage.")

    uploaded_file = st.file_uploader("Upload your Sleep Data From Last Night", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        sheet_name = "MeghaPatel"  # Replace with your Google Sheet name
        sheet = connect_to_google_sheets(sheet_name)

        # Handle Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        # Use OCR-based parsing
        structured_df = extract_structured_data_from_image(image)

        # Optionally use GPT to refine parsing
        gpt_parsed_df = use_gpt_for_parsing(text)
        if gpt_parsed_df is not None:
            st.write("Final Extracted Data:")
            st.dataframe(gpt_parsed_df)

        # Save to Google Sheets
        if sheet and gpt_parsed_df is not None and st.button("Save Image Data to Google Sheets"):
            save_to_google_sheets(sheet, gpt_parsed_df)
        
        # Usage in the app
        if selected_view == "Upload Data" and gpt_parsed_df is not None:
            # Extract sleep data from the image
            sleep_data = extract_structured_data_from_image(image)

            if not sleep_data.empty:
                st.write("### Extracted Sleep Data")
                st.dataframe(sleep_data)

                # Analyze sleep cycles
                st.write("### Sleep Cycle Analysis")
                try:
                    insights = analyze_sleep_cycles(sleep_data)
                    st.text(insights)
                except ValueError as e:
                    st.error(f"Error analyzing sleep data: {e}")
            else:
                st.error("No valid sleep data could be extracted from the image.")
                    # Generate and display sleep summary
            st.write("### Daily Sleep Summary")
            summary = generate_sleep_summary(gpt_parsed_df.iloc[0].to_dict())
            if summary:
                st.text(summary)

# **View 2: Daily Astrology**
elif selected_view == "Daily Astrology":
    st.title("Your Daily Astrology")
    st.write("Personalized insights for your day based on your data!")

    sheet_name = "MeghaPatel"  # Replace with your Google Sheet name
    sheet = connect_to_google_sheets(sheet_name)

    if sheet:
        try:
            rows = sheet.get_all_values()
            if rows:
                central_data = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming first row is headers

                # Generate personalized insights using OpenAI
                insights_prompt = f"""
                Using the following sleep data, generate a fun and personalized "daily astrology" insight:
                {central_data.tail(5).to_string(index=False)}
                """
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": insights_prompt}],
                )
                astrology_insight = response.choices[0].message.content
                st.write("\u2728 Your Daily Insight \u2728")
                st.text(astrology_insight)
            else:
                st.warning("No data found. Upload your data in the 'Upload Data' section.")
        except Exception as e:
            st.error(f"Failed to fetch or process data: {e}")

# **View 3: Deep Dive + Chat**
elif selected_view == "Deep Dive Analysis":
    st.title("Deep Dive Analysis")
    st.write("Analyze your aggregated data and interact with the AI assistant.")

    sheet_name = "MeghaPatel"  # Replace with your Google Sheet name
    sheet = connect_to_google_sheets(sheet_name)

    if sheet:
        try:
            rows = sheet.get_all_values()
            if rows:
                central_data = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming first row is headers

                # Show aggregated data
                st.write("### Aggregated Data")
                st.dataframe(central_data)

                # Generate detailed analysis using OpenAI
                analysis_prompt = f"""
                Provide an in-depth analysis of the following sleep data. Include trends, suggestions, and areas for improvement:
                {central_data.to_string(index=False)}
                """
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": analysis_prompt}],
                )
                detailed_analysis = response.choices[0].message.content
                st.write("### AI Analysis")
                st.text(detailed_analysis)

                # Chat Feature
                st.write("### Chat with AI Assistant")
                user_input = st.text_input("Ask a question about your data, supplements, or workouts:")
                if user_input:
                    chat_response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert health assistant."},
                            {"role": "user", "content": user_input},
                        ],
                    )
                    st.text(chat_response.choices[0].message.content)
            else:
                st.warning("No data found. Upload your data in the 'Upload Data' section.")
        except Exception as e:
            st.error(f"Failed to fetch or process data: {e}")
