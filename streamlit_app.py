import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd
import openai
import gspread
from pillow_heif import register_heif_opener
from google.oauth2.service_account import Credentials
import json
import re
import base64

# Register HEIF opener for PIL
register_heif_opener()

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

# Set the Tesseract path
if st.secrets["environment"]["TESSERACT_PATH"]:  # Use custom environment variable if set
    pytesseract.pytesseract.tesseract_cmd = st.secrets["environment"]["TESSERACT_PATH"]
else:  # Default path (local system)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
            model="gpt-4o-mini",
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
    
def parse_json_response(response_text):
    try:
        # Trim and locate JSON content
        json_string = response_text.strip()
        start_idx = json_string.find("{")
        end_idx = json_string.rfind("}") + 1
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in response.")

        # Extract the JSON
        json_string = json_string[start_idx:end_idx]

        # Parse JSON
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

def extract_sleep_data(image_path):
    """Extracts detailed sleep data including a table for analysis and saves to Google Sheets."""
    try:
        # Convert image to base64
        image_path.seek(0)  # Reset file pointer to the start
        img_base64 = base64.b64encode(image_path.read()).decode("utf-8")

        # Prepare the prompt for OpenAI
        prompt = [
            {
                "type": "text",
                "text": "Extract and structure the following sleep data into two parts: 1. Provide a detailed analysis and insights based on the data, including: - Overall Sleep Summary (highlight patterns, quality, and anomalies). - Efficiency and Quality (compare to recommended standards). - Blood Oxygen Trends (indicate concerns like dips or stability). - HRV Trends (highlight possible stress or recovery insights). - Heart Rate Trends (include any noteworthy patterns or changes). - Sleep Stage Trends and Comparison (identify balance among stages). - Correlations of Blood Oxygen, HRV, and Heart Rate over time. - Actionable recommendations for improvement where applicable. 2. A table with the following columns and 17 rows: -Date(YYYY-MM-DD) - Time (hh:mm) - Heart Rate (bpm) - Sleep Stage - Blood Oxygen (%) - HRV (ms). Return the data in JSON format, with keys 'summary' and 'table'. Ensure the summary is a coherent paragraph and the table is well-structured."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }
        ]

        # Request data extraction from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        extracted_data = response.choices[0].message.content.strip()
        # Parse the JSON response       
        data = parse_json_response(extracted_data)
        # Process summary data
        summary_text = data["summary"]
        # Process table data
        table_df = pd.DataFrame(data["table"])
        # Connect to Google Sheets
        sheet = connect_to_google_sheets("MeghaSleepChart")
        if not sheet:
            st.error("Failed to connect to Google Sheets.")
            return

        # Display and save summary data
        st.write("### Sleep Summary")
        st.markdown(summary_text)
        if st.button("Save Summary to Google Sheets"):
            save_to_google_sheets(sheet, pd.DataFrame([{"Summary": summary_text}]))

        # Display and save table data
        st.write("### Detailed Sleep Table")
        st.dataframe(table_df)
        if st.button("Save Table to Google Sheets"):
            success = save_to_google_sheets(sheet, table_df)
            if success:
                st.write("Data successfully saved.")
    except Exception as e:
        st.error(f"Error extracting or saving sleep data: {e}")


# Sidebar Navigation
st.sidebar.title("Navigation")
views = ["Upload Data", "Daily Astrology", "Deep Dive + Chat"]
selected_view = st.sidebar.radio("Go to", views)

# **View 1: Upload Data**
if selected_view == "Upload Data":
    st.title("Upload Sleep Data")
    st.write("Upload sleep data images for analysis and storage.")

    uploaded_file = st.file_uploader("Upload your Sleep Data From Last Night", type=["png", "jpg", "jpeg", "heic"])
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
            # Analyze sleep cycles
            st.write("### Sleep Cycle Analysis")
            try:
                insights = extract_sleep_data(uploaded_file)
                st.markdown(insights)
            except ValueError as e:
                st.error(f"Error analyzing sleep data: {e}")
        else:
            st.error("No valid sleep data could be extracted from the image.")
                    
# **View 2: Daily Astrology**
elif selected_view == "Daily Astrology":
    st.title("Your Daily Astrology")
    st.write("Personalized insights for your day based on your data!")

    sheet_name = "MeghaPatel"  # Replace with your Google Sheet name
    daily_astrology_sheet_name = "MeghaPatelAstrology"  # New sheet for storing astrology insights
    sheet = connect_to_google_sheets(sheet_name)
    astrology_sheet = connect_to_google_sheets(daily_astrology_sheet_name)

    if sheet and astrology_sheet:
        try:
            # Load existing data from the main sheet
            rows = sheet.get_all_values()
            if rows:
                central_data = pd.DataFrame(rows[1:], columns=rows[0])  # Assuming first row is headers

                # Check if astrology for today is already generated
                today_date = pd.Timestamp.now().strftime("%Y-%m-%d")
                existing_astrology = pd.DataFrame(astrology_sheet.get_all_values()[1:], columns=astrology_sheet.row_values(1))
                if not existing_astrology.empty and today_date in existing_astrology["Date"].values:
                    # Astrology for today already exists
                    today_astrology = existing_astrology.loc[existing_astrology["Date"] == today_date, "Daily Astrology"].iloc[0]
                    st.write("\u2728 Your Daily Insight for Today \u2728")
                    st.markdown(today_astrology)
                else:
                    # Generate personalized insights using OpenAI
                    insights_prompt = f"""
                    Using the following cumulative sleep data, generate a fun and personalized "daily astrology" insight
                    for today ({today_date}) with emojis:
                    {central_data.to_string(index=False)}
                    """
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": insights_prompt}],
                    )
                    astrology_insight = response.choices[0].message.content

                    # Display the generated astrology
                    st.write("\u2728 Your Daily Insight for Today \u2728")
                    st.markdown(astrology_insight)

                    # Save the astrology insight to the Google Sheet
                    astrology_sheet.append_row([today_date, astrology_insight])
                    st.success("Today's astrology insight has been saved!")
            else:
                st.warning("No data found. Upload your data in the 'Upload Data' section.")
        except Exception as e:
            st.error(f"Failed to fetch or process data: {e}")

# **View 3: Deep Dive Analysis**
elif selected_view == "Deep Dive Analysis":
    st.title("Deep Dive Analysis")
    st.write("Analyze your aggregated data and interact with the AI assistant.")

    # Google Sheet names
    sleep_metrics_sheet_name = "MeghaPatel"  # Main sleep metrics
    sleep_chart_sheet_name = "MeghaPatelSleepChart"  # Sleep cycle, HRV, Blood Oxygen, Heart Rate by time
    astrology_sheet_name = "MeghaPatelAstrology"  # Daily astrology insights

    # Connect to Google Sheets
    sleep_metrics_sheet = connect_to_google_sheets(sleep_metrics_sheet_name)
    sleep_chart_sheet = connect_to_google_sheets(sleep_chart_sheet_name)
    astrology_sheet = connect_to_google_sheets(astrology_sheet_name)

    # Initialize data frames
    sleep_metrics_data = pd.DataFrame()
    sleep_chart_data = pd.DataFrame()
    astrology_data = pd.DataFrame()

    # Fetch data with error handling
    try:
        if sleep_metrics_sheet:
            sleep_metrics_data = pd.DataFrame(
                sleep_metrics_sheet.get_all_values()[1:], 
                columns=sleep_metrics_sheet.row_values(1)
            )
        if sleep_chart_sheet:
            sleep_chart_data = pd.DataFrame(
                sleep_chart_sheet.get_all_values()[1:], 
                columns=sleep_chart_sheet.row_values(1)
            )
        if astrology_sheet:
            astrology_data = pd.DataFrame(
                astrology_sheet.get_all_values()[1:], 
                columns=astrology_sheet.row_values(1)
            )
    except Exception as e:
        st.warning("Failed to fetch data. Ensure Google Sheets are properly configured.")

    # If no data is available
    if sleep_metrics_data.empty and sleep_chart_data.empty and astrology_data.empty:
        st.warning("No data found. Please upload your sleep data in the relevant sections.")
    else:
        try:
            # Display Key Metrics
            st.write("### Key Metrics Overview")

            if not sleep_metrics_data.empty:
                total_sleep_hours = sleep_metrics_data["Total Duration"].str.extract(r'(\d+)').astype(float).sum()[0]
                average_sleep_score = sleep_metrics_data["Sleep Score"].astype(float).mean()
                st.metric("Total Sleep Hours (Last 7 Days)", f"{total_sleep_hours} hours")
                st.metric("Average Sleep Score", f"{average_sleep_score:.1f}")

            if not sleep_chart_data.empty:
                avg_blood_oxygen = sleep_chart_data["Blood Oxygen"].astype(float).mean()
                avg_hrv = sleep_chart_data["HRV"].astype(float).mean()
                avg_heart_rate = sleep_chart_data["Heart Rate"].astype(float).mean()
                st.metric("Average Blood Oxygen", f"{avg_blood_oxygen:.1f}%")
                st.metric("Average HRV", f"{avg_hrv:.1f} ms")
                st.metric("Average Heart Rate", f"{avg_heart_rate:.1f} bpm")

            # Visualization of Trends
            if not sleep_chart_data.empty:
                st.write("### Sleep Trends")
                st.line_chart(sleep_chart_data.set_index("Date")[["Heart Rate", "HRV"]])
                st.line_chart(sleep_chart_data.set_index("Date")[["Blood Oxygen"]])

            # AI Analysis
            st.write("### AI Analysis")
            analysis_prompt = f"""
            Using the following data, provide a detailed analysis of trends, anomalies, and recommendations for improving sleep and health metrics:
            Sleep Metrics:
            {sleep_metrics_data.to_string(index=False) if not sleep_metrics_data.empty else "No sleep metrics data available."}

            Sleep Chart Data:
            {sleep_chart_data.to_string(index=False) if not sleep_chart_data.empty else "No sleep chart data available."}

            Astrology Insights:
            {astrology_data.to_string(index=False) if not astrology_data.empty else "No astrology insights data available."}
            """
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
            )
            detailed_analysis = response.choices[0].message.content
            st.write("### AI-Generated Insights")
            st.markdown(detailed_analysis)

            # Chat Feature
            st.write("### Chat with AI Assistant")
            user_input = st.text_input("Ask a question about your sleep data or health:")
            if user_input:
                chat_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert health assistant analyzing sleep and health data."},
                        {"role": "user", "content": user_input},
                    ],
                )
                st.text(chat_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error during analysis or visualization: {e}")
