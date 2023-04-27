import cv2
import torch
import numpy as np
import math
from scipy import stats

def get_depth_map(filename):

    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    #cv2.imwrite("output.jpg", output)
    return output

def binarize_depth_map(depth_map):
    threshold = np.mean(depth_map) #set threshold
    bin_depth_map = np.where(depth_map > threshold, 1, 0) #binarize image
    return bin_depth_map

def map_objects_to_planes(annotations, depth_map):
    for obj in annotations['yolo'] + annotations['grit'] + annotations['craft']:
        x0, y0, x1, y1 = obj[1:-1]
        region = depth_map[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)]
        plane = stats.mode(region, axis=None, keepdims=True)[0]
        if plane:
            obj.append("foreground")
        else:
            obj.append("background")
    return annotations