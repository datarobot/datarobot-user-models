# Clustering Model with MLOps Reporting
Starting in DataRobot 9.1, unstructured models can adopt a target type and report feature and prediction information using the DataRobot MLOps agent which is automatically injected into the scoring hook. A more detailed example of this behavior is documented in [python3_unstructured_with_mlops_reporting](../python3_unstructured_with_mlops_reporting/). 

This example shows how a user can leverage the "Unstructured (Multiclass)" model type to support hosting clustering models in DataRobot. This approach requires that the user know the number of clusters in advance and configure the clusters as the targets for the multiclass model. 

This model is intended to be used with the Python3 SKlear Public Drop in Environment. 

The clusters use the IRIS [Dataset.](../..tests/testdata/iris_with_spaces_full.csv). 


## Calling this Model

DataRobot Unstructured custom models can accept any Mimetype as input. This tempalte model has been specifically designed to accept JSON data. The format is as follows: 

```
[
    {
        "Sepal.Length":5.1,
        "Sepal.Width":3.5,
        "Petal.Length":1.4,
        "Petal.Width":0.2
    },
    {
        "Sepal.Length":4.9,
        "Sepal.Width":3.0,
        "Petal.Length":1.4,
        "Petal.Width":0.2
    },
    {
        "Sepal.Length":4.7,
        "Sepal.Width":3.2,
        "Petal.Length":1.3,
        "Petal.Width":0.2
    },
    {
        "Sepal.Length":4.6,
        "Sepal.Width":3.1,
        "Petal.Length":1.5,
        "Petal.Width":0.2
    }
]
```

When this model is deployed you would call by sending this JSON String in the body of your POST request. 

### Note
Unstructued models cannot be used for batch predictions. 

