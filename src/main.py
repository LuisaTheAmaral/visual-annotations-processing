import json
from depth_perception import get_depth_map, binarize_depth_map, map_objects_to_planes
from plot_boxes import plotFilteredBoundingBoxes
from parse_annotations import merge_object_detections
from sentence_builder import find_groups, build_sentence
from categorize import categorize_annotations

f = open('annotations.json')
img_annotations = json.load(f)
f.close()

img_name = img_annotations["id"]

# create depth map of the image and binarize it
input_image = "../imgs/" + img_name
depth_map = get_depth_map(input_image)
bin_depth_map = binarize_depth_map(depth_map)

# use depth map to establish which objects are on the foreground and which are on the background
map_objects_to_planes(img_annotations["grit"], bin_depth_map)
map_objects_to_planes(img_annotations["yolo"], bin_depth_map)
map_objects_to_planes(img_annotations["craft"], bin_depth_map)

# filter object detections to eliminate redundant detections
filtered_object_detections = merge_object_detections([img_annotations["grit"], img_annotations["yolo"]])

#categorize annotations
categories = categorize_annotations(filtered_object_detections, img_annotations["craft"], img_annotations["places"], img_annotations["clipcap"])

#generate final full text sentence
sentence = build_sentence(filtered_object_detections, img_annotations["craft"], img_annotations["places"], img_annotations["clipcap"])    

print("-------------------------------------")
for cat in categories:
    print(f"{cat}:")
    print(categories[cat])

print("-------------------------------------")
print(sentence)
print("-------------------------------------")