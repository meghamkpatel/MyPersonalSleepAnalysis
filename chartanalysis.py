import cv2
import numpy as np
import pytesseract
import json

# Load the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    scale_percent = 50  # Reduce to 50% of original size for faster processing
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return resized_image, gray, edges

# Extract charts and structured data from the image
def extract_charts_and_data(resized_image, gray, edges, output_dir):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extracted_data = {}
    counter = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Threshold dimensions to capture only significant areas
            roi = resized_image[y:y+h, x:x+w]
            output_path = f"{output_dir}/chart_{counter}.png"
            cv2.imwrite(output_path, roi)

            roi_gray = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi_gray)
            extracted_data[f"chart_{counter}"] = {
                "text": text.strip(),
                "dimensions": (x, y, w, h)
            }

            if "heart rate" in text.lower():
                extracted_data[f"chart_{counter}"]["type"] = "Heart Rate Graph"
            elif "oxygen" in text.lower():
                extracted_data[f"chart_{counter}"]["type"] = "Blood Oxygen Graph"
            elif "sleep" in text.lower():
                extracted_data[f"chart_{counter}"]["type"] = "Sleep Stages Graph"
            elif "hrv" in text.lower():
                extracted_data[f"chart_{counter}"]["type"] = "HRV Graph"
            else:
                extracted_data[f"chart_{counter}"]["type"] = "Unknown"

            counter += 1
    return extracted_data, counter

# Save structured data to a JSON file
def save_structured_data(extracted_data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)

# Main function
def main(image_path, output_dir, output_json):
    resized_image, gray, edges = load_and_preprocess_image(image_path)
    extracted_data, counter = extract_charts_and_data(resized_image, gray, edges, output_dir)

    save_structured_data(extracted_data, output_json)

    print(f"Extracted {counter} potential charts/graphs. Saved to {output_dir} and structured data to {output_json}")

    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\Megha Patel\Downloads\Image-1 (1).jpg"  # Replace with your image path
    output_dir = r"C:\Users\Megha Patel\MyPersonalSleepAnalysis\charts"  # Replace with your output directory
    output_json = r"C:\Users\Megha Patel\MyPersonalSleepAnalysis\extracted_data.json"  # Replace with your output JSON file
    main(image_path, output_dir, output_json)
