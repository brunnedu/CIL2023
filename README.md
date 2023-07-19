# CIL Road Segmentation Project
This repository contains the code of our group project for the Computational Intelligence Lab (SS23 ETH Zürich).

### Team Members:
* Dustin Brunner
* Michael Ungersböck
* Siusing Yip
* Yanick Zengaffinen

## Setup
We recommend using the conda package manager. Navigate to the CIL2023 folder and run:\
`conda create --name cil python=3.8`\
`conda activate cil`\
`pip install -r requirements.txt`

#### Preprocessing:
Some models require prior extraction of masks (e.g. refinement, mask-flow-intersection-deadend approach etc). Make sure to run the corresponding notebooks or commands beforehand. 

#### Training:
1. Specify your train configuration in the `TRAIN_CONFIG` dictionary in `config.py`
2. Execute `python main.py train` in the command line.

In case the training crashes: Set `resume_from_checkpoint` to `True` in the `TRAIN_CONFIG` and specify the full `experiment_id` (including the timestamp!) of the training run you want to continue.

#### Testing:
Execute `python main.py run "path/to/test_data" "experiment_id"` in the command line.\
Make sure to fully specify the `experiment_id` (including the timestamp!).

#### Submission Generation:
`python main.py submission "experiment_id"` \
In case you want to specify a different foreground threshold, use `-t 0.5` for example.\
Make sure to fully specify the experiment name (including the timestamp!).

#### Refinement:
Model will be trained to learn the difference between the current mask and the actual mask (it also gets the original image as the input). Since this is an easier task than the original classification, it might perform better.\
1. Use the config that was used for the base model and update the run experiment id
1. Generate the low quality masks using `python main.py prepare-for-refinement`
1. Update the config to whatever you want, set IS_REFINEMENT = True
1. You can now run training/testing like for basic training

#### Votenet:
1. Run the ADW-Transform notebook on the training and validation data (adjust DATASET_FOLDER). This gives you the groundtruths of the 3 modalities (angle, distance, width)
2. Train a model for each of the 3 modalities (adjust the groundtruth_subfolder in config for each of them)
3. Generate the mean and std for all 3 trained models on the validation and test data (adjust data_dir and select_channel [0 = mean, 1 = std])
4. Run the ADW-Reconstruct notebook on the validation and test data. This gives you the mask priors.

#### Random:
Run `python main.py generate-random <in-dir> <out-dir>` to generate random masks.


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
