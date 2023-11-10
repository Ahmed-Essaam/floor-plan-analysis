import requests
import json
import imghdr

# Assuming 'images' is a list of image paths
images = [r"GCP-API\test\test2.jpg", r"GCP-API\test\test3.png"]

# Create a dictionary with the required structure
data = {"inputs": {"floor_plans": images, "roof_plan": r"GCP-API\test\1306-Roof.png"}}

# Create a list of tuples for the files
files = []

for i, image_path in enumerate(images):
    with open(image_path, 'rb') as file:
        file_content = file.read()
        file_type = imghdr.what(None, h=file_content)
        files.append(('floor_plans', (f'image_{i}', file_content, f'image/{file_type}')))

# Add the roof plan file
roof_plan_path = data["inputs"]["roof_plan"]
with open(roof_plan_path, 'rb') as roof_file:
    roof_file_content = roof_file.read()
    file_type = imghdr.what(None, h=roof_file_content)
    files.append(('roof_plan', (f'roof_plan', roof_file_content, f'image/{file_type}')))

# Send the POST request with files and data
resp = requests.post("http://127.0.0.1:5000/", files=files, data={"inputs": json.dumps(data)})

# Print the response
print(resp.json())
