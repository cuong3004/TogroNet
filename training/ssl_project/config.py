CONFIG = {
    "data": {
        "image_dir_train": "/home/agi/fiftyone/open-images-v7/train/data",
        "image_dir_valid": "/home/agi/fiftyone/open-images-v7/validation/data",
        "image_size": 256,
        # "val_ratio": 0.2,
    },

    "train": {
        "batch_size": 128,
        "num_workers": 4,
        "max_epochs": 500,
        "precision": "16-mixed",
        "learning_rate": 5e-4,
        "min_learning_rate": 1e-6,
        "warmup_epochs": 5,
        "weight_decay": 1e-6,
    },

    "ssl": {
        "z_dim": 2048,
        "lambda_coeff": 5e-3,
    },

    "system": {
        "accelerator": "gpu",
        "devices": 1,
    }
}
