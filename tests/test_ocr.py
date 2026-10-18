from app.ingestion.ocr import extract_text_from_image


image_path = "uploads/Capture.PNG"

text = extract_text_from_image(image_path)

print("OCR RESULT:")
print(text)