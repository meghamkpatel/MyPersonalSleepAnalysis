# Sleep Data Analysis and Astrology Insights Project

## Project Overview
This project is a Streamlit-based web application designed to help users analyze their sleep data, monitor health metrics, and receive personalized astrology insights. The app integrates data from Google Sheets, performs advanced analytics using AI models, and provides interactive visualizations and AI-powered assistance.

### Key Features
1. **Data Upload and Storage**:
   - Users can upload their sleep data (images or metrics).
   - The app extracts and structures data using OCR and AI-based parsing.
   - Data is saved in Google Sheets for further analysis.

2. **Daily Astrology Insights**:
   - Generates personalized daily astrology insights based on sleep data and health metrics.
   - Saves daily insights into a dedicated Google Sheet for future reference.

3. **Deep Dive Analysis**:
   - Displays key health metrics and visual trends for sleep cycles, heart rate, blood oxygen levels, and HRV.
   - Provides AI-generated insights and recommendations.

4. **Interactive Chat**:
   - Users can interact with an AI assistant to ask specific questions about their sleep data, health metrics, and lifestyle.

### Technology Stack
- **Frontend**: Streamlit for user interface and data visualization.
- **Backend**:
  - OpenAI API for AI-based text generation.
  - Python libraries for data manipulation and visualization (e.g., Pandas, Matplotlib).
- **Data Storage**: Google Sheets for storing sleep metrics, cycle data, and astrology insights.
- **OCR**: Tesseract OCR for extracting text from images.

## Application Structure

### 1. **Data Upload**
- **Functionality**:
  - Upload sleep data as images or metrics.
  - Extract structured data using OCR and parse it with AI models.
- **Storage**:
  - Data is saved to a Google Sheet for further processing.

### 2. **Daily Astrology**
- **Functionality**:
  - Combines cumulative sleep data to generate daily astrology insights.
  - Checks for existing insights for the current date to avoid duplication.
- **Storage**:
  - Saves astrology insights to a dedicated Google Sheet.

### 3. **Deep Dive Analysis**
- **Functionality**:
  - Displays key metrics:
    - Total Sleep Hours
    - Average Sleep Score
    - Average Blood Oxygen Levels
    - Average Heart Rate Variability (HRV)
    - Average Heart Rate
  - Generates visual trends for sleep stages, heart rate, and blood oxygen.
  - Provides AI-driven insights and recommendations.

### 4. **Interactive Chat**
- **Functionality**:
  - Users can ask health-related questions and get AI-powered responses.

## AI Models

### Prerequisites
1. Python 3.8+
2. Google Cloud project with a service account.
3. Tesseract OCR installed locally:
   - Windows: Install from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - Linux/Mac: Use package manager (`apt`, `brew`).

## Future Enhancements
1. **Enhanced AI Models**:
   - Explore fine-tuned models for improved insights.
2. **Mobile Compatibility**:
   - Optimize the app for mobile devices.
3. **Additional Metrics**:
   - Incorporate metrics like calorie intake or activity levels.
4. **Customization**:
   - Allow users to customize astrology insights (e.g., add preferred metrics).



## License
This project is licensed under the MIT License. See the LICENSE file for details.

TODO LIST:
- ~~analyze charts and graphs from images~~
- ~~fix google credentials to be in streamlit~~
- ~~save daily horoscope~~
- add data
- test data analysis 
- add emojis and aesthetic colors
- incorporate actual astrology
