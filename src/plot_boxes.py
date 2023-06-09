import cv2
import json
from random import randrange
from os import listdir
from os.path import isfile, join

def plotBoundingBoxes(img_path, output_name, annotations, color=None, threshold=0.55):

    img = cv2.imread(img_path)

    for obj, x0, y0, x1, y1, score in annotations:
        if score >= threshold:
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            if not color:
                color = getRandomColor()
            cv2.rectangle(img, start_point, end_point, color=color, thickness=2)

            cv2.putText(
                img,
                obj,
                (int(x0), int(y0) - 10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = color,
                thickness=2
            )
    
    cv2.imwrite(output_name, img)

def plotFilteredBoundingBoxes(img_path, output_name, annotations):

    img = cv2.imread(img_path)

    for obj, x0, y0, x1, y1, score, plane in annotations:
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        if plane == "foreground":
            color = (0,0,255)
        else:
            color = (255,0,0)
        cv2.rectangle(img, start_point, end_point, color=color, thickness=2)

        cv2.putText(
            img,
            obj,
            (int(x0), int(y0) - 10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = color,
            thickness=2
        )
    
    cv2.imwrite(output_name, img)

def getRandomColor():
    return (randrange(255), randrange(255), randrange(255))

if __name__ == "__main__":
    
    annotations_path = 'annotations2.json'
    f = open(annotations_path)
    annotations = json.load(f)
    f.close()

    images = [f for f in listdir('../imgs/') if isfile(join('../imgs/', f))]
    for img in annotations:
        input_name = '../imgs/' + img["id"]
        output_name = "../annotated/" + img["id"]

        plotBoundingBoxes(input_name, output_name, img['yolo'], color=(0,0,255))
        plotBoundingBoxes(output_name, output_name, img['grit'], color=(0,255,0), threshold=0.55)
        plotBoundingBoxes(output_name, output_name, img['craft'], color=(255,0,255))

        # img = cv2.imread(output_name)
        # cv2.imshow(output_name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
    
    