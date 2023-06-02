import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import math

def _tokenize_and_tag(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return tagged

# Define a function to compute the similarity ratio between two object descriptions using WordNet
def _compute_similarity_ratio(word1, word2):
    synset1 = wn.synsets(word1)
    synset2 = wn.synsets(word2)
    if synset1 and synset2:
        path_sim = max(s1.path_similarity(s2) for s1 in synset1 for s2 in synset2)
        return path_sim if path_sim is not None else 0
    else:
        return 0

def get_max_similarity(str1, str2):
    tagged1 = _tokenize_and_tag(str1)
    tagged2 = _tokenize_and_tag(str2)

    nouns1 = [word.lower() for word, tag in tagged1 if tag.startswith('N')]
    nouns2 = [word.lower() for word, tag in tagged2 if tag.startswith('N')]

    ratio = 0
    for noun1 in nouns1:
        for noun2 in nouns2:
            ratio = max(ratio, _compute_similarity_ratio(noun1, noun2))
    return ratio

#returns ratio of overlap of two bounding boxes
def get_overlap_ratio(bb1, bb2):
    xmin, ymin, xmax, ymax, _, plane1 = bb1
    xmin2, ymin2, xmax2, ymax2, _, plane2 = bb2
    overlap = []
    if not (xmin > xmax2 or xmin2 > xmax) or not (ymax > ymin2 or ymax2 > ymin) and plane1 == plane2:
        # boxes overlap
        # area of intersection
        SI = max(0, min(xmax, xmax2) - max(xmin, xmin2)) * max(0, min(ymax, ymax2) - max(ymin, ymin2))
        # area of the union
        SA = (xmax - xmin)*(ymax - ymin)
        SB = (xmax2 - xmin2)*(ymax2 - ymin2)
        SU = SA + SB - SI
        # ratio of overlap
        ratio = SI / SU
        return ratio
    return 0

#reduces redundancy in object detections by checking bounding boxes that overlap
def merge_object_detections(object_detections):
    
    #annotations related to object detection
    for annotations in object_detections:
        if len(annotations):
            # Compute the overlap ratios and similarity ratios between elements in annotations
            for i, obj1 in enumerate(annotations):
                keep_obj1 = True
                keep_obj2 = True
                for j, obj2 in enumerate(annotations[i+1:]):
                    j = i + j + 1  # adjust index for nested loop
                    keep_obj2 = True
                    if len(obj1) > 2 and len(obj2) > 2:
                        overlap_ratio = get_overlap_ratio(obj1[1:], obj2[1:])
                        similarity = get_max_similarity(obj1[0], obj2[0])
                    else: #one of the objects or both dont't have bounding box
                        overlap_ratio = 0
                        similarity = 0
                    if overlap_ratio >= 0.7 or (0.4 <= overlap_ratio < 0.7 and similarity > 0.5):
                        if obj1[-1] >= obj2[-1]:
                            if obj2 in annotations:
                                annotations.remove(obj2)
                        else:
                            if obj1 in annotations:
                                annotations.remove(obj1)
                                break #object 1 is out so no need to compare it with the remaining objects

    #specific for MEMORIA integration
    grit = object_detections[0]
    remove_grit_stop_words(grit)
    yolo = object_detections[1]
    
    merged_list = []
    if len(yolo) and len(grit):
        # Compute the overlap ratios and similarity ratios between elements in yolo and grit
        overlap_ratios = np.zeros((len(yolo), len(grit)))
        similarity_ratios = np.zeros((len(yolo), len(grit)))
        for i, elem1 in enumerate(yolo):
            for j, elem2 in enumerate(grit):
                if len(elem1)> 2 and len(elem2) > 2:
                    overlap_ratios[i, j] = get_overlap_ratio(elem1[1:], elem2[1:])
                    similarity_ratios[i, j] = get_max_similarity(elem1[0], elem2[0])
                else:
                    overlap_ratios[i, j] = 0
                    similarity_ratios[i, j] = 0

        # Filter out elements from yolo that have an overlap ratio greater than or equal to 0.7 with any element in grit
        merged_list = [yolo[i] for i in range(len(yolo)) if np.max(overlap_ratios[i]) < 0.7]

        # Filter out elements from yolo that have an overlap ratio between 0.4 and 0.7 with at least one element in grit and that have a high object similarity
        for i, elem1 in enumerate(yolo):
            for j, elem2 in enumerate(grit):
                if 0.4 <= overlap_ratios[i, j] < 0.7 and similarity_ratios[i, j] < 0.5:
                    merged_list.append(elem1)
                    break
    elif len(yolo):
        return yolo
    elif len(grit):
        return grit

    # Combine the filtered yolo and grit lists to create the final merged list
    return merged_list + grit

def remove_grit_stop_words(annotations):
    stop_words = ['a', 'an', 'the']
    
    for det in annotations:
        det[0] = ' '.join([word for word in det[0].split() if word not in stop_words])

