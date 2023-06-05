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
    non_overlapped_objs = obj_detections[:]
    
    for ocr in ocr_detections:
        overlaps = []
        for obj in obj_detections:
            if len(ocr) > 2 and len(obj) > 2:
                overlap = get_ocr_overlap(ocr[1:], obj[1:])
            else:
                overlap = 0
            if overlap > 0.5:
                overlaps.append( (obj, overlap) )

        #if the ocr overlapped with multiple detections, determine that it is written in the one with the highest ratio
        if overlaps:
            max_overlaps = [x[0] for x in overlaps if x[1] == max(overlaps, key = lambda x: x[1])[1]]
            #if there are multiple detections with the highest ratio, determine that it is written in the one with the smallest area
            chosen_obj = min(max_overlaps, key = lambda x: (x[3] - x[1])*(x[4] - x[2]))            
            
            ocr_obj_mapping[chosen_obj[0]].append(ocr[0])
            if chosen_obj in non_overlapped_objs:
                non_overlapped_objs.remove(chosen_obj)
        else: #if ocr did not overlap with any object, store it in a list and add later to the final string
            misc.append(ocr[0])
        
    for obj in ocr_obj_mapping:
        for ocr in ocr_obj_mapping[obj]:
           s += f"{ocr}, " 
        s = s[:-2]
        s += f" on {obj}. "

    if misc:
        for ocr in misc:
            s += f"{ocr}, "
        s = s[:-2]
        s += ". "

    return s, non_overlapped_objs
    
def parse_objects(detections):

    foreground = []
    background = []
    user_added = []

    for det in detections:
        if len(det) == 2:
            user_added.append(det[0])
        elif det[-1] == "foreground":
            foreground.append(det)
        else:
            background.append(det)

    #sort by confidence
    foreground = sorted(foreground, key= lambda x: x[-2], reverse=True)
    background = sorted(background, key= lambda x: x[-2], reverse=True)

    #after sorting store only the object names
    foreground = [obj[0] for obj in foreground]
    background = [obj[0] for obj in background]
    
    txt = ' '.join(user_added + foreground + background)

    return f"{txt}. "

def parse_places(detections):
    s = ""

    categories = list(detections["categories"].items())
    categories = sorted(categories, key = lambda x: x[1], reverse=True)

    for det, _ in categories:
        s += f"{det.replace('_', ' ').replace('/', ' ')} "

    for det in detections["tags"]:
        det = det.replace('_', ' ').replace('/', ' ')
        s += f"{det} "

    return s

def build_sentence(objects, ocr, places, descriptions, locations):
    
    s = ""
    for desc in descriptions:
        s += desc + " "
    
    ocr_sentence, reduced_objects = parse_ocr(objects, ocr)
    obj_sentence = parse_objects(reduced_objects)
    places_sentence = parse_places(places)

    return s + obj_sentence + ocr_sentence + places_sentence + " ".join(locations)
