import gspread
from google.oauth2.service_account import Credentials

# Define the API scope
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", 
         "https://www.googleapis.com/auth/drive"]

file = r"c:\Users\Megha Patel\Downloads\sleepproject-447516-46e613673714.json"

# Load credentials from the JSON file
CREDS = Credentials.from_service_account_file(file, scopes=SCOPE)

# Authenticate with Google Sheets
gc = gspread.authorize(CREDS)

# Open the Google Sheet
sheet = gc.open("MeghaPatel").sheet1  # Replace with the actual name of your sheet

# Write a test row to the sheet
sheet.append_row(["Test", "Row", "Data"])

# Print data from the sheet
all_data = sheet.get_all_values()
print("Sheet Data:", all_data)
