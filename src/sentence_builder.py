from collections import defaultdict

def find_groups(detections):
    groups = []
    for i in range(len(detections)):
        group = [detections[i]]
        for j in range(len(detections)):
            if i != j:
                # check if object i's bounding box contains object j's bounding box
                if (detections[i][1] <= detections[j][1] and
                    detections[i][2] <= detections[j][2] and
                    detections[i][3] >= detections[j][3] and
                    detections[i][4] >= detections[j][4] and
                    detections [i][-1] == detections[j][-1]):
                    group.append(detections[j])
        if len(group) > 1:
            groups.append(group)

    return _merge_groups(groups)

def _merge_groups(lists):
    merged_lists = []
    while len(lists) > 0:
        merged_list = lists.pop(0)
        i = 0
        while i < len(lists):
            if any(obj in merged_list for obj in lists[i]):
                merged_list.append(lists.pop(i))
            else:
                i += 1
        merged_lists.append(list(merged_list))
    return merged_lists

def detections_group_to_sentence(group):
    group = sorted(group, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
    biggest = group[0]

#returns ratio of overlap of two bounding boxes
def get_ocr_overlap(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If the boxes do not intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the areas of the two bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the ratio of the area of box1 inside box2
    iou = intersection_area / box1_area
    
    return iou

def parse_ocr(obj_detections, ocr_detections):
    
    s = ""
    ocr_obj_mapping = defaultdict(list) #stores associations with objects and ocr that overlaps with them
    misc = [] #stores ocr that does not overlap with any object
    
    for ocr in ocr_detections:
        overlaps = []
        for obj in obj_detections:
            overlap = get_ocr_overlap(ocr[1:], obj[1:])
            if overlap > 0.7:
                overlaps.append(obj)

        #if the ocr overlapped with multiple detections, determine that it is written in the one with the smallest area
        if overlaps:
            chosen_obj = min(overlaps, key = lambda x: (x[3] - x[1])*(x[4] - x[2]))
            ocr_obj_mapping[chosen_obj[0]].append(ocr[0])
        else: #if ocr did not overlap with any object, store it in a list and add later to the final string
            misc.append(ocr[0])
        
    for obj in ocr_obj_mapping:
        s += f"There is "
        for ocr in ocr_obj_mapping[obj]:
           s += f"'{ocr}', " 
        s = s[:-2]
        s += f" written in {obj}. "

    if misc:
        s += "The words "
        for ocr in misc:
            s += f"'{ocr}', "
        s = s[:-2]
        s += " can also be read in the scene. "

    return s

    

    
