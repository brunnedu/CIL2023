# CIL Road Segmentation Project
This repository contains the code of our group project for the Computational Intelligence Lab (SS23 ETH Zürich).

## Setup
We recommend using the conda package manager. Navigate to the CIL2023 folder and run:\
`conda create --name cil python=3.8`\
`conda activate cil`\
`pip install -r requirements.txt`

#### Training:
1. specify your train configuration in the TRAIN_CONFIG dictionary in config.py
2. execute `python main.py train` in the command line while being in the root directory of the project.

In case the training crashes: Set resume_from_checkpoint to True and specify the full experiment_id (including the auto-appended timestamp!) of the training run you want to continue in TRAIN_CONFIG.

#### Testing:
`python main.py run "path to test data" "experiment name"` \
Make sure to fully specify the experiment name (including the timestamp!).

#### Submission Generation:
`python main.py submission "experiment name"` \
In case you want to specify a different foreground threshold, use `-t 0.5` for example.\
Make sure to fully specify the experiment name (including the timestamp!).

#### To get a comprehensive list of all possible configurations: 
Run `python main.py conf` and look at the generated markdown schema.

#### To see a list of all available commands type:
`python main.py --help`

## Image Collection
Additional training images can be collected using the `src/image_collection.py` script.
The script requires a valid Google Maps API key in the `GMAPS_API_KEY` environment variable.
Usage can be checked using `python src/image_collection.py --help`

### Location Config
The location config acts as an input to the image collection process.
The `bbox` property describes the bounding box of the specific location.
The first value corresponds to the upper-left corner and the second to the lower-right corner. 
