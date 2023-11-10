from floorPlans_helperFunctions import *
from Basic_FloorPlan_Testcase import get_roof_outerlines
import traceback
import logging
from flask import Flask, request, jsonify
import json

''' 
### Description: The function Detects floor plan objects in uploaded images using a pre-trained model 
                and returns a list of dictionaries containing information about the detected objects.
### Inputs: 
    - files_uploaded: a list of uploaded image np matrix files in any format supported by the Python Imaging Library (PIL).
### Outputs:
    - coors_info: A list of dictionaries containing information about the detected floor plan objects for all uploaded images 
    (e.g. walls, windows and doors starting, ending points within the thickness).
'''
def get_Image_Seg(raw_image, device, split, model):
    coors_info = []
    try:
        BGR_fplan, scale = read_image(raw_image)
        # Get the floor plan detection results
        polygons, types = process_image(BGR_fplan, device, model, split)
        print(polygons)
        
        # window_outliers_thresh = get_window_outliers(polygons, types, scale)
        for plt_coors in get_coors_for_comparision(polygons, types, scale):
            coors_info.append(plt_coors)

    except Exception as e:
        logging.error(f"Error in get_Image_Seg: {e}")
        traceback.print_exc()
        # You might want to return a default value or handle the error differently
        # return None

    return coors_info

from pathlib import Path
# load the model
file_path = Path(__file__).resolve().parent
device, split, model = load_model(file_path)

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        # Load JSON data from the request or default to an empty dictionary
        json_data = json.loads(request.form.get("inputs") or '{}')

        # Retrieve 'inputs' key from JSON data with a default value of an empty dictionary
        inputs = json_data.get('inputs', {})

        # Retrieve 'floor_plans' and 'roof_plan' from 'inputs' with default values
        floor_plans = inputs.get('floor_plans', [])
        roof_plan = inputs.get('roof_plan')


        # Check if both 'floor_plans' and 'roof_plan' are present
        if not floor_plans and not roof_plan:
            raise ValueError("Missing floor plans and roof plan")

        # Process each floor plan file
        floor_coors_info = []
        for i, file in enumerate(request.files.getlist("floor_plans")):
            print(f"Received Floor Plan {i + 1}:", file.filename)
            coors_info = get_Image_Seg(file, device, split, model)
            floor_coors_info.append(coors_info)
        print("GOT FLOOR PLAN INFO")

        # Process the roof plan file
        roof_plan_file = request.files.get("roof_plan")
        print("roof_plan_file", roof_plan_file)
        print("Received Roof Plan:", roof_plan_file.filename)
        roof_polygons = get_roof_outerlines(roof_plan_file)
        print("GOT ROOF PLAN INFO")
        # Process the roof plan file as needed

        return jsonify({ "floor_plans": floor_coors_info, "roof_plan": roof_polygons })


    except (ValueError, KeyError, json.JSONDecodeError) as e:
        # Handle specific exceptions and return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle other exceptions and return a 500 Internal Server Error response
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)