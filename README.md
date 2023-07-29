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

## Datasets

We are using the following datasets:
1. [data5k](https://drive.google.com/file/d/1oEQxTkbbR6IGRzjAWxWGvl5ypW5RzuLW/view?usp=drive_link): training dataset (2 x 5'000 images)
2. [data30k](https://drive.google.com/file/d/1oNNIm0GIxr3GM5TkKnDY_OvsWqVW3k9e/view?usp=drive_link): alternative training dataset (2 x 30'000 images)
3. [training (from kaggle)](https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2023/data): validation dataset (144 images)
4. [test500](https://drive.google.com/file/d/1iXyVD5-aFIm66LsndtDpds89LR3qG771/view?usp=drive_link): test dataset (2 x 1'000 images)
5. [test](https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2023/data): submission dataset (144 images)

## Reproduce Results
This is a step-by-step guide on how to reproduce our results. Reoccuring steps will be explained below.
Before starting: download all data and ideally put them into a ./data folder (otherwise you'll have to adjust the paths in the configs).
Ensemble experiments require you to be on the `dustin/ensemble-precomputed` branch.

#### _Random_
1. Run `python main.py generate-random data/test500 out/random/test500` to generate random masks.
2. Run & Evaluate on _test500_
3. Run & Submit _test_

### _U-Net++ R152 60k_ and _D-LinkNet R152 60k_
1. Select the corresponding config from ./report/configs and paste it into ./config.py
2. Train on _data30k_
3. Run & Evaluate on _test500_
4. Run & Submit _test_

### _Ensemble Baseline_
Requires _U-Net++ R152 60k_, _D-LinkNet R152 60k_ and 3 models each with only R50 backbone analogously (see ./report/configs/submodels)
1. Run all submodels on following datasets: _training_, _test_, _test500_, _data5k_
2. Train Ensemble based on submodel predictions (see ./report/configs/ensemble_baseline.py)
3. Run & Evaluate Ensemble on _test500_
4. Run & Submit _test_

### _VoteNet R50 10k_
1. Run the ADW-Transform notebook on the training and validation data (adjust DATASET_FOLDER). This gives you the groundtruths of the 3 modalities (angle, distance, width)
2. Train a model for each of the 3 modalities (adjust the groundtruth_subfolder in config for each of them) (see ./report/configs)
3. Generate the mean and std for all 3 trained models on the validation and test data (adjust data_dir and select_channel [0 = mean, 1 = std] in config)
4. Run the ADW-Reconstruct notebook on _test500_ and _test_ data. This gives you the mask priors.
5. Evaluate on _test500_
6. Submit _test_

### _Ensemble incl. VoteNet_
Requires _U-Net++ R152 60k_, _D-LinkNet R152 60k_ and 3 models each with only R50 backbone analogously (see ./report/configs).
Additionally requires _VoteNet R50 10k_.
1. Run all submodels on following datasets: _training_, _test_, _test500_, _data5k_ (for VoteNet this implies transformation, running and reconstruction)
2. Train Ensemble based on submodel predictions (see ./report/configs/ensemble_baseline.py)
3. Run & Evaluate Ensemble on _test500_
4. Run & Submit _test_

### _Refined Ensemble incl. VoteNet_
Requires a trained _Ensemble incl. VoteNet_
1. Generate the low quality masks using `python main.py prepare-for-refinement` (equivalent to calling _run_ on all data)
2. Update config (see ./report/configs for the specifics). Notably IS_REFINEMENT = True
3. Train on _data5k_
4. Optional: Repeat steps 1 & 3 arbitrarily often
4. Run & Evaluate on _data500_
5. Run & Submit _test_

## Individual Steps
### Training
Trains a model and chooses the model with lowest validation loss.
1. Specify your model-, data- and train-configuration in the `TRAIN_CONFIG` dictionary in `config.py`
2. Execute `python main.py train` in the command line
In case the training crashes: Set `resume_from_checkpoint` to `True` in the `TRAIN_CONFIG` and specify the full `experiment_id` (including the timestamp!) of the training run you want to continue.

### Run
Generates the outputs of a trained model on a specified set of images.
1. Specify your run configuration in the `TRAIN_CONFIG` dictionary in `config.py`. Make sure the model config is consistent with the config used during training.
2. Execute `python main.py run` in the command line
In case you prefer to use the last instead of the best model checkpoint, append a -l (see main.py for all options)

### Evaluate
Computes Loss and F1-score (including std dev over all samples) for a specific model. Make sure you have run the model first or at least setup the proper config so it can run automatically!
1. `python main.py evaluate` 

### Submission
Generates submission masks and the submission.csv in the format specified by the kaggle competition.
1. `python main.py submission "experiment_id"` (include timestamp in experiment_id)
In case you want to specify a different foreground threshold, use `-t 0.5` for example.

### To see a list of all available commands type:
`python main.py --help`

## Image Collection
Additional training images can be collected using the `src/image_collection.py` script.
The script requires a valid Google Maps API key in the `GMAPS_API_KEY` environment variable.
Usage can be checked using `python src/image_collection.py --help`

### Location Config
The location config acts as an input to the image collection process.
The `bbox` property describes the bounding box of the specific location.
The first value corresponds to the upper-left corner and the second to the lower-right corner. 
