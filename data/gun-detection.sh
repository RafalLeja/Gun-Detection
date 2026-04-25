#!/bin/bash
mkdir sources
if [ -f ./sources/gun-detection.zip ]; then
    echo "File already exists. Skipping download."
else
    echo "Downloading gun detection dataset..."
    curl -L -o ./sources/gun-detection.zip https://www.kaggle.com/api/v1/datasets/download/ugorjiir/gun-detection
fi
unzip ./sources/gun-detection.zip -d ./sources