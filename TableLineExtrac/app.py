from flask import Flask, request, render_template, send_file
import os
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for model and processor
model = None
processor = None
ang_model = None

def initialize_models():
    global model, processor, ang_model
    
    # Initialize Qwen2VL model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Initialize processor
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    # Initialize Doctr OCR model
    ang_model = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="parseq",
        pretrained=True,
        det_bs=8,
        reco_bs=1024,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
    )

def remove_blur(image):
    variance_of_laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    
    if variance_of_laplacian < 100:
        print("Image is blurry, applying deblurring technique.")
        deblurred_image = wiener_deconvolution(image)
        return deblurred_image
    else:
        return image
    
def wiener_deconvolution(image):
    kernel = np.ones((5, 5), np.float32) / 25
    deconvolved = cv2.filter2D(image, -1, kernel)
    return deconvolved

def set_image_dpi(image):
    PIL_image = Image.fromarray(image)
    dpi_image_path = "dpi_adjusted_image.png"
    PIL_image.save(dpi_image_path, dpi=(300, 300))
    return dpi_image_path

def preprocess_image(ang_model, image_path):
    try:
        doc = DocumentFile.from_images([image_path])
        result = ang_model(doc)
        json_res = result.export()
        orientation = json_res["pages"][0]["orientation"]["value"]
        print(f"Detected orientation: {orientation} degrees")
    except Exception as e:
        print(f"Error detecting orientation using Doctr: {str(e)}")
        orientation = 0

    image = cv2.imread(image_path)

    if orientation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = remove_blur(image)
    resized = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
    if len(resized.shape) != 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)
    
    cleaned = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
    cleaned_image = Image.fromarray(cleaned)
    preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], "cleaned_output_image.png")
    cleaned_image.save(preprocessed_path, format="PNG")
    return preprocessed_path

def process_invoice_image(image_path, model, processor):
    cleaned_image_path = preprocess_image(ang_model, image_path)
    image = Image.open(cleaned_image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": """
                    You are given a hospital invoice which can be in any languge, it may contain multiple tables, each representing a different category. For each table, your task is to:

                    1. Identify and retrieve the category name of the table.
                                - If there is only one table or no category name is provided, use "Not Applicable" for the 
                                  category name.
                    2. Extract the list of items and their corresponding final amounts.                                                        
                                 - If both quantity and rate of item is provided for an item, calculate the final amount as:
                                   Final Amount = Rate of item * Quantity
                    The output format should be strictly as follows:
                    Category: {category name}
                    Item: {item name}
                    Final Amount: {Final Amount}
                    Repeat this format for each table step by step in the invoice. Ensure that all items and their final amounts are correctly matched to their respective categories.Give the required output in the same language as input.
    """
                }
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def extract_items_and_amounts(text):
    text = text.replace("\n", " ").strip()
    results = []
    category_totals = {}
    current_category = None
    
    lines = text.split("Category:")

    def clean_amount(amount_str):
        # First, try to find a standalone number
        import re
        numerical_match = re.search(r'\b\d+(?:,\d{3})*(?:\.\d{2})?\b', amount_str)
        if numerical_match:
            amount_str = numerical_match.group()
        
        # Remove any currency symbols, commas, and extra text
        cleaned = ''.join(c for c in amount_str if c.isdigit() or c == '.' or c == '-')
        try:
            return float(cleaned)
        except ValueError:
            print(f"Could not convert amount to float: {amount_str}")
            return 0.0

    for line in lines[1:]:
        if "Item:" in line and "Final Amount:" in line:
            try:
                category = line.split("Item:")[0].strip()
                item_amount_pairs = line.split("Item:")[1:]
                
                # If we're switching to a new category, add the total for the previous one
                if category != current_category:
                    if current_category and current_category in category_totals:
                        results.append({
                            "category": current_category,
                            "item_name": "CATEGORY TOTAL",
                            "item_amount": f"{category_totals[current_category]:.2f}"
                        })
                    current_category = category
                
                if category not in category_totals:
                    category_totals[category] = 0.0

                for pair in item_amount_pairs:
                    item_name = pair.split("Final Amount:")[0].strip()
                    amount_str = pair.split("Final Amount:")[1].strip()
                    
                    # Clean and extract the numerical amount
                    amount = clean_amount(amount_str)
                    category_totals[category] += amount
                    
                    # Format the amount consistently
                    formatted_amount = f"{amount:.2f}"
                    
                    results.append({
                        "category": category,
                        "item_name": item_name,
                        "item_amount": formatted_amount
                    })
            
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line}. Error: {e}")
                continue
    
    # Add the total for the last category
    if current_category and current_category in category_totals:
        results.append({
            "category": current_category,
            "item_name": "CATEGORY TOTAL",
            "item_amount": f"{category_totals[current_category]:.2f}"
        })
    
    return results
# Update the upload_file route to handle the styled DataFrame
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            output = process_invoice_image(filepath, model, processor)
            results = extract_items_and_amounts(output)
            
            df = pd.DataFrame(results)
            
            # Style the DataFrame to highlight category totals
            def highlight_totals(row):
                if row['item_name'] == 'CATEGORY TOTAL':
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            
            styled_df = df.style.apply(highlight_totals, axis=1)
            
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_data.csv')
            df.to_csv(csv_path, index=False)
            
            return render_template('result.html', 
                                 tables=[styled_df.to_html(classes='data', header="true")], 
                                 titles=df.columns.values)
    
    return render_template('upload.html')

@app.route('/download')
def download_file():
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_data.csv')
    return send_file(csv_path, as_attachment=True, download_name='extracted_data.csv')

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True,port=5002)

