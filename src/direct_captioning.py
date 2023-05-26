#this script was initially used to join every annotations without processing them

import json
from collections import defaultdict
import inflect

p = inflect.engine()

class Description():

    def __init__(self, annotations) -> None:
        self.annotations = annotations

    def describe(self):
        s = ""
        s += self.getSceneContext()
        s += self.getSmallDescription()
        s += self.checkObjects()
        s += self.listDetailedObjects()
        s += self.checkOCR()
        return s

    def getSceneContext(self):
        places = self.annotations["places"]
        return f"This scene takes place {places['environment']}. "

    def checkOCR(self):
        ocr_detections = self.annotations["craft"]
        
        s = ""
        for ocr, xmin, ymin, xmax, ymax, score in ocr_detections:
            overlaps = self.checkBoundingBoxOverlap(xmin, ymin, xmax, ymax)
            if len(overlaps) == 1:
                s += f"There is a {overlaps[0]} where it is written '{ocr}.'"
            elif len(overlaps) > 1:
                obj, _ = max(overlaps, key=lambda x:x[1])
                s += f"There is a {obj} where it is written '{ocr}'."
            else:
                s+= f"'{ocr}' can be read."
        return s
    
    def checkObjects(self):
        s = "It features "
        d = defaultdict(int)
        for obj, xmin, ymin, xmax, ymax, score in self.annotations["yolo"]:
            d[obj] += 1

        for obj in d:
            if d[obj] > 1:
                s += f"{p.number_to_words(d[obj])} {p.plural(obj)}, "
            else:
                s += f"{p.number_to_words(d[obj])} {obj }, "
        return s[:-2] + ". "
    
    def listDetailedObjects(self):
        s = "In the picture there is "
        for obj, xmin, ymin, xmax, ymax, score in self.annotations['grit'][:-1]:
            if obj[:2] not in ['an', 'a ']:
                if obj[0] in ['a', 'e', 'i', 'o', 'u']:
                    s += "an "
                else:
                    s += "a "
            s += f"{obj}, "

        s = s[:-2] #remove last comma
        s += f" and {self.annotations['grit'][-1][0]}. "
        return s

    def checkBoundingBoxOverlap(self, xmin, ymin, xmax, ymax):
        objects = self.annotations['grit'] + self.annotations['yolo']
        overlap = []
        for object, xmin2, ymin2, xmax2, ymax2, _ in objects:
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
                overlap.append( (object, ratio) )
        return overlap
    
    def getSmallDescription(self):
        return f"This picture shows {self.annotations['clipcap'][1]} "
                
if __name__ == "__main__":
    
    f = open('annotations.json')
    annotations = json.load(f)
    f.close()
    d = Description(annotations)
    print(d.describe())
