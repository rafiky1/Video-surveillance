import cv2
import time
import os
import datetime
import numpy as np
from google.cloud import vision, storage
from dotenv import load_dotenv

# Load API Key and set up Google Cloud authentication
load_dotenv()
GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_NAME = "your-gcp-bucket-name"  # Replace with your GCP bucket name

if not GCP_CREDENTIALS:
    raise ValueError("Google Cloud Credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS in .env")

# Initialize Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

# Create a directory for local image storage
IMAGE_DIR = "captured_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def capture_image():
    """Captures an image from the webcam and saves it locally."""
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # Ensure the camera initializes
    ret, frame = cap.read()
    
    if not ret:
        raise IOError("Cannot capture frame from webcam.")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{IMAGE_DIR}/capture_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    
    cap.release()
    return filename

def upload_to_gcp(file_path):
    """Uploads an image to Google Cloud Storage."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(os.path.basename(file_path))
    blob.upload_from_filename(file_path)
    print(f"Uploaded {file_path} to Google Cloud Storage.")

def detect_objects(image_path):
    """Detects objects in an image using Google Vision API."""
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = vision_client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    detected_objects = {obj.name: obj.score for obj in objects if obj.score > 0.50}
    
    return detected_objects

def image_difference(img1, img2, threshold=5):
    """Compares two images to detect movement or changes."""
    image1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:
        return True  # Consider them different if dimensions don't match

    diff = cv2.absdiff(image1, image2)
    non_zero_count = np.count_nonzero(diff)

    return (non_zero_count / diff.size) * 100 > threshold

def main():
    previous_image = None
    
    while True:
        print("\nCapturing new image...")
        image_path = capture_image()

        # Detect objects in the captured image
        detected_objects = detect_objects(image_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if detected_objects:
            print(f"Objects detected at {timestamp}: {detected_objects}")
        else:
            print(f"No objects detected at {timestamp}")

        # If it's the first image or a significant change is detected, upload it
        if previous_image is None or image_difference(previous_image, image_path):
            upload_to_gcp(image_path)
            previous_image = image_path
            print(f"Image uploaded to GCP at {timestamp}.")
        else:
            print(f"No significant change detected at {timestamp}.")

        time.sleep(30)  # Wait 30 seconds before capturing the next image

if __name__ == "__main__":
    main()
