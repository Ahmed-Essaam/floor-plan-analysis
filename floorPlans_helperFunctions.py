# from skimage import transform
import numpy as np
import torch
import torch.nn.functional as F
from floortrans.models import get_model
from floortrans.loaders import RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons 
# from mpl_toolkits.axes_grid1 import AxesGrid
rot = RotateNTurns()
import cv2
from PIL import Image, ImageEnhance
import io
# model setup 
def load_model(path):
    print("Start")
    model = get_model('hg_furukawa_original',path, 51)
    n_classes = 44
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f'{path}/floortrans/models/model_best_val_loss_var.pkl', map_location=device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    print("Model loaded.")
    return device, split, model

# getting inputs 
def read_image(file_storage):
    binary_data = file_storage.read()
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(binary_data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Now rgb_image is in BGR format
    np_image = np.array(image)
    print("Number of channels:", np_image.shape[2])
    if np_image.shape[2] == 1:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    # resize image
    resized, aspect = resize_floorPlan(np_image)
    # flip the image ( make origin point bottom left)
    # resized = resized[::-1,:,:]
    print("Input size = ", resized.shape)
    return resized, aspect

# Resize Image
def resize_floorPlan(BGR_fplan):
    input_height, input_width = 512, 512
    print("Original image shape: " + str(BGR_fplan.shape) + " in H, W, C format")
    original_h, original_w =  BGR_fplan.shape[:2]
    aspect = original_w / float(original_h)
    print("Original aspect ratio: " + str(aspect))

    if aspect > 1:
        # Landscape orientation - wide image
        res = int(aspect * input_height)
        scale_ratio = input_height / original_h

        BGR_fplan = cv2.resize(BGR_fplan, (res, input_height), interpolation=cv2.INTER_AREA)

    if aspect < 1:
        # Portrait orientation - tall image
        res = int(input_width / aspect)
        scale_ratio = input_width / original_w

        BGR_fplan = cv2.resize(BGR_fplan, (input_width, res), interpolation=cv2.INTER_AREA)

    if aspect == 1:
        scale_ratio = 1.0
        BGR_fplan = cv2.resize(BGR_fplan, (input_width, input_height), interpolation=cv2.INTER_AREA)

    print("Scaled Image shape: " + str(BGR_fplan.shape))
    
    return BGR_fplan, scale_ratio

# Basic preprocessing (Brightness, Contrast, Sharpness)
def Image_PreProcessing(BGR_fplan):
    ## Filter by range gray color
    hsv = cv2.cvtColor(BGR_fplan, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 187])
    upper = np.array([179, 0, 255])
    mask = cv2.inRange(hsv, lower, upper)
    HSV_filter = cv2.bitwise_and(BGR_fplan, BGR_fplan, mask=mask)
    ## Basic Enhancments
    # convert image into gray scale    
    gray_fplan = cv2.cvtColor(HSV_filter, cv2.COLOR_BGR2GRAY)
    # Convert grayscale image to PIL Image object
    gray_pil = Image.fromarray(gray_fplan)
    # Image brightness enhancer
    enhancer = ImageEnhance.Brightness(gray_pil)
    brightness = enhancer.enhance(0.7)
    # Image Contrast enhancer
    enhancer2 = ImageEnhance.Contrast(brightness)
    Contrast = enhancer2.enhance(0.8)
    # Image Sharpness enhancer
    enhancer3 = ImageEnhance.Sharpness(Contrast)
    Sharpness = enhancer3.enhance(2)
    ## Model Requirements
    # convert image into array for further calculations 
    enhanced_img = np.array(Contrast)
    # correct color channels
    RGB_fplan = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)  
    # correct the dimension oreder Move from (h,w,3)--->(3,h,w) as model input dimension 
    RGB_fplan = np.moveaxis(RGB_fplan, -1, 0) 
    # Normalization values to range -1 and 1
    RGB_fplan = 2 * (RGB_fplan / 255.0) - 1  
    # Convert NumPy array to Pytorch tensor
    model_input_img = torch.FloatTensor(RGB_fplan) 
    # add extra dim  
    model_input_img = model_input_img.unsqueeze(0) 
    print("Image Preprocessing has done")
    
    return model_input_img

# getting segmentation 
def networks_segmentaion(device, model, img, n_classes = 44):
    with torch.no_grad():

        #Check if shape of image is odd or even
        size_check = np.array([img.shape[2],img.shape[3]])%2

        height = int(img.shape[2] - size_check[0])
        width = int(img.shape[3] - size_check[1])
        img_size = (height, width)
        
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width]).to(device)
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(img, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    print("networks_segmentaion has done")
    return prediction, img_size

# enhance these segmentations 
def post_processed_polygons(prediction, img_size, split):
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    polygons, types = get_polygons((heatmaps, rooms, icons), 0.15, [1, 2])
    print("post_processed_polygons has done")
    return polygons, types

# calculate if a line horizontal or vertical
def calc_line_dim(point_1, point_2):
    if abs (point_2[0] - point_1[0]) > abs (point_2[1] - point_1[1]):
        # horizontal
        line_dim = 'h'
    else:
        # vertical
        line_dim = 'v'
    return line_dim

# calculate other information for each line
def calc_line_info(point_1, point_2, point_3, point_4):
    # CALC THICKNESS
    thickness = np.linalg.norm(point_2 - point_1) 
    # Convert Unit 
    
    # CALC CENTER LINE  
    start_point = (point_1 + point_2) / 2 
    end_point = (point_3 + point_4) / 2 
    # Reformat it 
    start_point = {'X': start_point[0], 'Y': start_point[1], 'Z' : 0}
    end_point = {'X': end_point[0], 'Y': end_point[1], 'Z' : 0}

    return start_point, end_point, thickness

# takes a list of start points and a list of end points in the form of tuples, and calculates the length of the line segment between each pair of points
def calc_pixel_val_len(P1_keys, p2_keys):
    lengths = []
    for start, end in zip(P1_keys, p2_keys):
        length = np.linalg.norm(np.array(end) - np.array(start))
        lengths.append(length)

    return lengths

def fix_coordinates(coors_info, pixel_per_feet, origin_point): 
    
    # Fix Coordinates 
    X_in_pixel, Y_in_pixel = origin_point
    for key, value in coors_info.items():
        for item in value:
            # Convert the start and end points of the line to real-life measurements
            item['Startpoint']['X'] = (item['Startpoint']['X'] - X_in_pixel) / pixel_per_feet
            item['Startpoint']['Y'] = (item['Startpoint']['Y'] - Y_in_pixel) / pixel_per_feet
            item['Endpoint']['X'] = (item['Endpoint']['X'] - X_in_pixel) / pixel_per_feet
            item['Endpoint']['Y'] = (item['Endpoint']['Y'] - Y_in_pixel) / pixel_per_feet
            item['Thickness'] = item['Thickness'] / pixel_per_feet  
    
    return coors_info

def get_icon_outliers(polygons, types, scale, icon):
    print("scale", scale)
    mean_icon_len = []
    for i, pol in enumerate(polygons):
        point_1, point_2, point_3, point_4 = pol / scale

        if types[i]['class'] == icon:
            line_dim = calc_line_dim(point_1, point_3)
            if line_dim == 'h' : 
                length = np.linalg.norm(point_1 - point_2)
            elif line_dim == 'v': 
                length = np.linalg.norm(point_1 - point_4)
            mean_icon_len.append(length) 
    if not mean_icon_len:
        return np.Inf
    # Allocate the outliers using IQR 
    q1, q3 = np.percentile(mean_icon_len, [25, 75])
    print("q1",q1)
    print("q3",q3)
    iqr = q3 - q1
    print("iqr",iqr)
    upper_bound = q3 + (3 * iqr)
    print("upper_bound",upper_bound)
    icon_outliers = [x for x in mean_icon_len if x > upper_bound]
    icon_outliers_thresh = min(icon_outliers) if icon_outliers != [] else 10000
    print("icon_outliers", icon_outliers)
    return icon_outliers_thresh
   
def get_coors_for_comparision(polygons, types, scale):
    # icon = 1 if window, icon = 2 if door
    window_outliers_thresh = get_icon_outliers(polygons, types, scale, icon = 1)

    ## Create List of the following format [start_point, end_point, thickness, length] 
    for i, pol in enumerate(polygons):

        ## GET POINTS
        point_1, point_2, point_3, point_4 = pol / scale

        # Check Line Dimention to perform the suitable calculation on it based on if it's a horizontal or vertical line
        line_dim = calc_line_dim(point_1, point_3)
        
        ## DO THE CALCULATIONS 
        if line_dim == 'h' : 
            start_point, end_point, thickness = calc_line_info(point_1, point_4, point_3, point_2)
        elif line_dim == 'v': 
            start_point, end_point, thickness = calc_line_info(point_1, point_2, point_3, point_4)

        ## GET TYPE AND MERGE ALL OF THE INFORMATION
        if types[i]['type'] == 'wall':
            type = 'wall'
            yield {'Type': type, 'Startpoint' :start_point, 'Endpoint': end_point, 'Thickness': thickness}
        else: # Icons
            if types[i]['class'] == 2:
                type = 'door'
                yield {'Type': type, 'Startpoint' :start_point, 'Endpoint': end_point, 'Thickness': thickness}

            elif types[i]['class'] == 1:
                # garage condition 
                length = np.linalg.norm(np.array((start_point['X'], start_point['Y'])) - np.array((end_point['X'], end_point['Y'])))
                if length >= window_outliers_thresh:
                    type = 'door'
                else:
                    # general condition 
                    type = 'window'
                yield {'Type': type, 'Startpoint' :start_point, 'Endpoint': end_point, 'Thickness': thickness}
    print("Calculations has done")

# put all together
def process_image(input_img, device, model, split):
    enhanced_gray_fplan = Image_PreProcessing(input_img)
    prediction, img_size = networks_segmentaion(device, model, enhanced_gray_fplan, n_classes = 44)
    polygons, types = post_processed_polygons(prediction, img_size, split)

    return polygons, types