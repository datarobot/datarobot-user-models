## Python Unstructured Inference Model Template + MLOps

### Description

#### Model Information
* Target type: Unstructured (Binary)
* Target: complication
* Positive value: 1, Negative: 0

#### Model Files
* custom.py
* model.pkl


#### Datasets for Testing
* Training dataset: dataset/training-surgical-dataset.csv
* Holdout dataset: dataset/holdout-surgical-dataset.csv
* Actuals dataset: dataset/holdout-surgical-dataset.csv
* Prediction request dataset: dataset/predict-request-surgical-dataset.csv

  ##### Notes
  * Do not upload any of the datasets to the model's assembly. They are not part of the model
    itself, Instead, you may upload them to the AI Catalog or whenever required directly from the
    local file system.

#### Training
* The `train-and-generate-artifacts` folder contains a simple script to generate the `model.pkl`
  along with the datasets in the `datasets` folder.

  ##### Notes
  * Do not upload any of files in that folder to the model's assembly. They are not part of the
    model itself.
