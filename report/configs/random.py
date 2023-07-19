from src.metrics import FocalLoss, PatchAccuracy, PatchF1Score, ThresholdBinaryF1Score

TRAIN_CONFIG = {} # use generate-random command in main.py

RUN_CONFIG = {
    'experiment_id': 'random',
    'eval_metrics': [
        FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
        PatchF1Score(patch_size=16, cutoff=0.5),
        ThresholdBinaryF1Score(cutoff=0.5),
        PatchAccuracy(patch_size=16, cutoff=0.5),
    ],
}
