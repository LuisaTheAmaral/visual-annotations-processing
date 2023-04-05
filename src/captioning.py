import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

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
    xmin, ymin, xmax, ymax = bb1
    xmin2, ymin2, xmax2, ymax2 = bb2
    overlap = []
    if not (xmin > xmax2 or xmin2 > xmax) or not (ymax > ymin2 or ymax2 > ymin):
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
def merge_object_detections(annotations):
    #annotations related to object detection
    grit = annotations['grit']
    yolo = annotations['yolo']

    # Compute the overlap ratios and similarity ratios between elements in yolo and grit
    overlap_ratios = np.zeros((len(yolo), len(grit)))
    similarity_ratios = np.zeros((len(yolo), len(grit)))
    for i, elem1 in enumerate(yolo):
        for j, elem2 in enumerate(grit):
            overlap_ratios[i, j] = get_overlap_ratio(elem1[1:-1], elem2[1:-1])
            similarity_ratios[i, j] = get_max_similarity(elem1[0], elem2[0])

    # Filter out elements from yolo that have an overlap ratio greater than or equal to 0.7 with any element in grit
    merged_list = [yolo[i] for i in range(len(yolo)) if np.max(overlap_ratios[i]) < 0.7]

    # Filter out elements from yolo that have an overlap ratio between 0.4 and 0.7 with at least one element in grit and that have a high object similarity
    for i, elem1 in enumerate(yolo):
        for j, elem2 in enumerate(grit):
            if 0.4 <= overlap_ratios[i, j] < 0.7 and similarity_ratios[i, j] < 0.5:
                merged_list.append(elem1)
                break

    # Combine the filtered yolo and grit to create the final merged list
    return merged_list + grit

if __name__ == "__main__":
    
    f = open('annotations.json')
    annotations = json.load(f)
    f.close()
    
    print("Object detections filtered: ")
    for e in merge_object_detections(annotations):
        print(e[0])


    