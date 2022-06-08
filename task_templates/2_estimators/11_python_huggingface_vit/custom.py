import os
from pathlib import Path

from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


import pandas as pd
import numpy as np
import torch
from img_utils import b64_to_img
from datarobot_drum.custom_task_interfaces import BinaryEstimatorInterface


class DataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomTask(BinaryEstimatorInterface):
    def fit(self, X, y, row_weights=None, **kwargs):
        self.model_name = "vit-base-patch16-224"
        self.pretrained_location = Path(__file__).resolve().parent / self.model_name
        self.class_order_to_lookup(kwargs["class_order"])
        # load base transformer featurizer and model
        self.extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_location)
        estimator = AutoModelForImageClassification.from_pretrained(
            self.pretrained_location,
            num_labels=len(kwargs["class_order"]),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        # Create a training dataset with pytorch
        # Images are in b64 encoded format, and have to be turned into Image objects and then encoded using the encoder
        train_encoded = self.extractor(
            images=X.iloc[:, 0].apply(b64_to_img).values.tolist(), return_tensors="pt"
        )
        train_dataset = DataSet(train_encoded, y.map(lambda v: int(self.label2id[str(v)])))

        # Setup training arguments and the Huggingface trainer to facilitate fine tuning
        training_args = TrainingArguments(
            output_dir=kwargs["output_dir"] + "/training_tmp", num_train_epochs=3, no_cuda=True,
        )
        trainer = Trainer(model=estimator, args=training_args, train_dataset=train_dataset,)
        trainer.train()
        self.estimator = trainer

    def class_order_to_lookup(self, class_order):
        # Fine tuning a Huggingface estimator requires having a mapping from label to id and from id to label
        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(class_order):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label
        print("Label to ID mapping: ", self.label2id)

    def save(self, artifact_directory):
        """
        Serializes the object and stores it in `artifact_directory`

        Parameters
        ----------
        artifact_directory: str
            Path to the directory to save the serialized artifact(s) to.

        Returns
        -------
        self
        """

        # If your estimator is not pickle-able, you can serialize it using its native method,
        # i.e. in this case for Hugginface we use save_model, and then set the estimator to none
        self.estimator.save_model(Path(artifact_directory) / "vit")
        # # Helper method to handle serializing, via pickle, the CustomTask class
        self.save_task(artifact_directory, exclude=["estimator"])

        return self

    @classmethod
    def load(cls, artifact_directory):
        """
        Deserializes the object stored within `artifact_directory`

        Returns
        -------
        cls
            The deserialized object
        """
        custom_task = cls.load_task(artifact_directory)
        custom_task.estimator = AutoModelForImageClassification.from_pretrained(
            Path(artifact_directory) / "vit"
        )
        custom_task.extractor = AutoFeatureExtractor.from_pretrained(
            Path(artifact_directory) / custom_task.model_name
        )

        return custom_task

    def predict_proba(self, X, **kwargs):
        assert len(X.columns) == 1, "This only works with a single image column"
        # Transform the B64 encoded string into images that the model can work with
        images = X.iloc[:, 0].apply(b64_to_img).values.tolist()
        batch_size = 5
        size = len(images)
        preds = None
        # Batch up the predictions to keep memory usage lower
        for idx in range(0, size, batch_size):
            img_slice = images[idx : idx + batch_size]
            inputs = self.extractor(images=img_slice, return_tensors="pt")
            outputs = self.estimator(**inputs)
            # Get the predicted probability as numpy values
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()
            if preds is None:
                preds = predictions
            else:
                preds = np.vstack((preds, predictions))
        df = pd.DataFrame(preds, columns=self.label2id.keys())
        return df
