# visual-annotations-processing

## Description

This repository contains code that processes annotations obtained using various computer vision models. The purpose of this code is to streamline and enhance the annotation data by eliminating redundancy and organizing it into meaningful categories. The code categorizes the annotation content into different categories: optical characters recognized (OCR), places, attributes, and objects.

The processing pipeline includes identifying the importance of each object detection by determining in which image plane they belong to (foreground or background), checking the overlap between bounding boxes of object detections in the same plane and checking the overlap between OCR and object detections to infer in which objects the OCR text is written in.

The annotations can be extracted using the folowing [annotation toolbox](https://github.com/LuisaTheAmaral/image-annotation-tools).

## Installation:

```bash
conda env create -f environment.yml
conda activate env

python -m spacy download en_core_web_sm
```

## Usage:

Ensure that you have the necessary data and annotations in the correct format. The code expects input annotations in a specific format, so make sure your data follows the required structure. An example of this structure is presented in the file [```src/annotations2.json```](src/annotations2.json)

Run the main script to execute the annotation processing:

```bash
cd src
python3 main.py
```

The scripts outputs a categorization of the annotations into the categories:
- Places
- Objects
- OCR
- Attributes

It also outputs a string with all annotations concatenated, with added detail of in which object the detected OCR annotations are written in. Additionaly, the annotations appear in the string in the order of their importance of the analysis of the image. Annotations that are semantically richer or appear in the foreground of the image appear first.
