import spacy

nlp = spacy.load("en_core_web_sm")

def categorize_objects(detections, categories):
    for det in detections:
        categories["objects"].add(det[0])
    
def categorize_ocr(detections, categories):
    for det in detections:
        categories["ocr"].add(det[0])

def categorize_places(detections, categories):
    for det in detections["tags"]:
        det = det.replace('_', ' ').replace('/', ' ')
        categories["attributes"].add(det)

    for det, _ in detections["categories"].items():
        det = det.replace('_', ' ').replace('/', ' ')
        categories["places"].add(det)

def categorize_description(descriptions, categories):
    for desc in descriptions:
        doc = nlp(desc)

        for token in doc:
            if token.pos_ == "NOUN":
                categories["objects"].add(token.text)
            elif token.pos_ == "ADJ":
                categories["attributes"].add(token.text)

        for ent in doc.ents:
            if ent.label_ == "GPE":
                categories["places"].add(ent.text)

def categorize_annotations(objects, ocr, places, descriptions):
    
    categories = {
        "places": set(),
        "objects": set(),
        "attributes": set(),
        "ocr": set()
    }

    categorize_objects(objects, categories)
    categorize_ocr(ocr, categories)
    categorize_places(places, categories)
    categorize_description(descriptions, categories)

    return categories
    