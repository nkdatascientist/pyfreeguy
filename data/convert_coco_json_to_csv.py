from tqdm import tqdm 
import json
import csv

# https://stackoverflow.com/questions/62251509/coco-json-file-to-csv-format-path-to-image-jpg-x1-y1-x2-y2-class-name
# Load the COCO JSON file
with open('./COCO2014/annotations/instances_train2014.json', 'r') as f:
    coco_json = json.load(f)

# Create a CSV file and write the header
with open('coco.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
    
    # Loop through all annotations in the COCO JSON file
    for annotation in tqdm(coco_json['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        bbox_x = bbox[0]
        bbox_y = bbox[1]
        bbox_w = bbox[2]
        bbox_h = bbox[3]
        
        # Write the annotation data to the CSV file
        writer.writerow([image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h])
