import json
from depth_perception import get_depth_map, binarize_depth_map, map_objects_to_planes
from plot_boxes import plotFilteredBoundingBoxes
from parse_annotations import merge_object_detections
from sentence_builder import parse_ocr
from categorize import categorize_annotations

f = open('annotations.json')
img_annotations = json.load(f)
f.close()

img_name = img_annotations["id"]

# create depth map of the image and binarize it
# input_image = "../imgs/" + img_name
# depth_map = get_depth_map(input_image)
# bin_depth_map = binarize_depth_map(depth_map)

num_grit = len(img_annotations["grit"])
num_yolo = len(img_annotations["yolo"])

# use depth map to establish which objects are on the foreground and which are on the background
# map_objects_to_planes(img_annotations["grit"], bin_depth_map)
# map_objects_to_planes(img_annotations["yolo"], bin_depth_map)
#map_objects_to_planes(img_annotations["craft"], bin_depth_map)

# filter object detections to eliminate redundant detections
num_yolo_updated, num_grit_updated, filtered_object_detections = merge_object_detections([img_annotations["grit"], img_annotations["yolo"]])

ocr_overlaps = parse_ocr(filtered_object_detections, img_annotations["craft"])

with open("counts.csv", "a") as f:
    f.write(f"\n{num_grit}, {num_yolo}, {num_grit_updated}, {num_yolo_updated}, {ocr_overlaps}")


#categorize annotations
# categories = categorize_annotations(filtered_object_detections, img_annotations["craft"], img_annotations["places"], img_annotations["clipcap"], img_annotations["locations"])

#generate final full text sentence
# sentence = build_sentence(filtered_object_detections, img_annotations["craft"], img_annotations["places"], img_annotations["clipcap"], img_annotations["locations"])    

# print("-------------------------------------")
# for cat in categories:
#     print(f"{cat}:")
#     print(categories[cat])

# print("-------------------------------------")
# print(sentence)
# print("-------------------------------------")

