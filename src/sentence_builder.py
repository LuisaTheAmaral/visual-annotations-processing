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

    


    
