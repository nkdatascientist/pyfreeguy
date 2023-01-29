
#!/bin/bash

# Define the URLs for the COCO datasets and annotations
train_url="http://images.cocodataset.org/zips/train2014.zip"
val_url="http://images.cocodataset.org/zips/val2014.zip"
annotations_url="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# Define the local paths to save the datasets and annotations
train_path="train2014.zip"
val_path="val2014.zip"
annotations_path="annotations_trainval2014.zip"

cd ../../data/
if [ -d "./COCO2014" ]; then
    echo "COCO2014 already exists, skipping."
else
    mkdir COCO2014
fi
cd COCO2014

train_directory_path="$(pwd)/train2014"
train_file_path="$(pwd)/train2014.zip"
val_directory_path="$(pwd)/val2014"
val_file_path="$(pwd)/val2014.zip"
annotaion_directory_path="$(pwd)/annotations"
annotaion_file_path="$(pwd)/annotations_trainval2014.zip"

# Download and Extract the datasets. Clean up the downloaded zip files
function df_ {
    if [ -f "$1" ]; then
        echo "$1 already exists."
    else
        echo "$1 does not exist, creating now."
        wget -O $3 $4
    fi
    if [ -d "$2" ]; then
        echo "$2 already exists."
    else
        echo "$2 does not exist, creating now."
        unzip $1
    fi
}
df_ $train_file_path $train_directory_path $train_path $train_url
df_ $val_file_path $val_directory_path $val_path $val_url
df_ $annotaion_file_path $annotaion_directory_path $annotations_path $annotations_url

echo "successfully downloaded the coco dataset"
cd ./research/nishanth/