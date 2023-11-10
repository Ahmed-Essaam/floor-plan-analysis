import cv2 
import numpy as np 
from PIL import Image, ImageEnhance
import io
import easyocr
import pandas as pd
from scipy.spatial import cKDTree


def preprocessing_img(file_storage):
    binary_data = file_storage.read()
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(binary_data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Now rgb_image is in BGR format
    np_image = np.array(image)
    # convert the image to grayscale format
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    # blured Image 
    blur_img = cv2.bilateralFilter(gray,9,75,75)
    # apply binary thresholding
    thresh = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    print("preprocessing_img is done")
    return thresh, gray
    
def get_outer_contour(thresh):
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
            
    # Contour filtering to get largest area
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area > max_area:
            max_area = area
            largest_contour = c
    # Approximate the contour to obtain the exterior points
    epsilon = 0.001 * cv2.arcLength(largest_contour, closed=True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
    # Extract the exterior points from the approximation
    exterior_points = np.array(approx[:, 0, :]) 
    print("get_outer_contour is done")
   
    return exterior_points

def reformat_points(exterior_points):
    # # Get start and end points for each line segment
    n_points = len(exterior_points)
    roof_lines = []
    for i in range(n_points):
        start_point = exterior_points[i]
        end_point = exterior_points[(i + 1) % n_points]
        # Reformat it 
        start_point = {'X': int(start_point[0] ), 'Y': int(start_point[1]), 'Z' : 0}
        end_point = {'X': int(end_point[0] ), 'Y': int(end_point[1]), 'Z' : 0}
        # Save it
        line = {'Startpoint' :start_point, 'Endpoint': end_point, 'Thickness': None}
        roof_lines.append(line)
    print("reformat_points is done")

    
    return roof_lines

def perform_ocr(gray, language):
    # Enhance Image Contrast
    gray_pil = Image.fromarray(gray)
    enhancer2 = ImageEnhance.Contrast(gray_pil)
    Contrast = enhancer2.enhance(2)
    contrast_np = np.array(Contrast)

    # Initialize bounds before the try block
    bounds = None

    # Perform OCR on the resized image
    reader = easyocr.Reader([language])
    try:
        bounds = reader.readtext(contrast_np, width_ths=0.7, paragraph=True, rotation_info=[90, 180, 270])
    except Exception as e:
        print(f"Error in perform_ocr: {e}")

    # Check if bounds is not None before creating DataFrame
    if bounds is not None:
        bounds = pd.DataFrame(bounds).rename(columns={0: 'Coordinates', 1: 'text'})
        print("perform_ocr is done")
    else:
        bounds = pd.DataFrame(columns=['Coordinates', 'text'])

    return bounds

def filter_ocr(bounds):
    numeric_bounds = bounds[
        bounds["text"].apply(
            lambda x: any(i.isnumeric() for sublist in x for i in sublist)
            and not any(i.isalpha() for sublist in x for i in sublist)
        )
    ]
    return numeric_bounds

def add_nearest_text_info(text_df, roof_lines, k=3):
    # Extract the midpoints and convert to numpy array
    text_coords = np.array([np.mean(np.array(box), axis=0) for box in text_df['Coordinates']])
    tree = cKDTree(text_coords)

    for line in roof_lines:
        start_point = np.array([line['Startpoint']['X'], line['Startpoint']['Y']])
        end_point = np.array([line['Endpoint']['X'], line['Endpoint']['Y']])

        # Calculate midpoint
        midpoint = (start_point + end_point) / 2

        # Query the KD-tree for nearest neighbors
        distances, nearest_indices = tree.query(midpoint, k=k)

        # Filter out-of-bounds indices
        valid_indices = [index for index in nearest_indices if 0 <= index < len(text_df)]

        # Create a list of nearest_texts using valid indices
        nearest_texts = [
            {'text': text_df.iloc[index]['text'], 'Coordinates': text_df.iloc[index]['Coordinates']}
            for index in valid_indices
        ]
        # Add NearestText and Distances to the line dictionary
        if nearest_texts:
            line['NearestText'] = nearest_texts

    return roof_lines

def convert_to_serializable(item):
    if isinstance(item, (np.ndarray, list)):
        return [convert_to_serializable(subitem) for subitem in item]
    elif isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif isinstance(item, (int, float, str)):
        return item
    else:
        return str(item)
    
def get_roof_outerlines(binary_data): 
    try:
        thresh, gray = preprocessing_img(binary_data)
        exterior_points = get_outer_contour(thresh)
        roof_lines = reformat_points(exterior_points)
        bounds = perform_ocr(gray, 'en')
        numeric_bounds = filter_ocr(bounds)
        modified_roof_lines = add_nearest_text_info(numeric_bounds, roof_lines)
        print("modified_roof_lines", modified_roof_lines)
        # Convert any nested NumPy arrays to lists
        roof_polygons_serializable = convert_to_serializable(modified_roof_lines)

    except Exception as e:
        print(f"Error in get_roof_outerlines: {e}")
        return roof_lines
    return roof_polygons_serializable



