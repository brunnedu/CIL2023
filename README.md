# CIL Road Segmentation Project
This repository contains the code of our group project for the Computational Intelligence Lab (SS23 ETH Zürich).

## Setup
We recommend using the conda package manager. Navigate to the CIL2023 folder and run:\
`conda create --name cil python=3.8`\
`conda activate cil`\
`pip install -r requirements.txt`

#### You can then train the model by executing:
`python main.py train "path to training data" -id "experiment name"` \
In case the training crashes and you want to resume from the last checkpoint, use the `-r` flag.

#### You can then run the model on some test data by executing:
`python main.py run "path to test data" "experiment name"`

#### You can then generate a submission by executing:
`python main.py submission "experiment name"` \
In case you want to specify a different foreground threshold, use `-t 0.5` for example.

To see a list of all available commands type: \
`python main.py --help`

## Image Collection
Additional training images can be collected using the `src/image_collection.py` script.
The script requires a valid Google Maps API key in the `GMAPS_API_KEY` environment variable.
Usage can be checked using `python src/image_collection.py --help`

### Location Config
The location config acts as an input to the image collection process.
The `bbox` property describes the bounding box of the specific location.
The first value corresponds to the upper-left corner and the second to the lower-right corner. 
