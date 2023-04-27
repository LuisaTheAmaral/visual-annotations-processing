import json
from depth_perception import get_depth_map, binarize_depth_map, map_objects_to_planes
from plot_boxes import plotFilteredBoundingBoxes
from parse_annotations import merge_object_detections
from sentence_builder import find_groups

f = open('annotations.json')
annotations = json.load(f)
f.close()

for img_annotations in annotations:
    img_name = img_annotations["id"]

    aux = []
    for i in img_annotations["grit"]:
        if i[5] >= 0.55:
            aux.append(i)
    img_annotations["grit"] = aux

    # create depth map of the image and binarize it
    input_image = "../imgs/" + img_name
    depth_map = get_depth_map(input_image)
    bin_depth_map = binarize_depth_map(depth_map)

    # use depth map to establish which objects are on the foreground and which are on the background
    map_objects_to_planes(img_annotations, bin_depth_map)

    # filter object detections to eliminate redundant detections
    filtered_object_detections = merge_object_detections(img_annotations)
    #print(filtered_object_detections)

    del img_annotations["grit"]
    del img_annotations["yolo"]
    img_annotations["object_detection"] = filtered_object_detections

    #print(find_groups(img_annotations["object_detection"]))

    # with open("filtered_annotations.json", "w") as outfile:
    #     json.dump(annotations, outfile)

    # output_name = "../annotated/filtered_" + img_name
    # plotFilteredBoundingBoxes(input_image, output_name, img_annotations['object_detection']) 