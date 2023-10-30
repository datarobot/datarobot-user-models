# Clustering Model as Multiclass

This example shows how a user can leverage the "Multiclass" model type to support hosting clustering models in DataRobot. This approach requires that the user know the number of clusters in advance and configure the clusters as the targets for the multiclass model. 

This model is intended to be used with the Python3 SKlearn Public Drop in Environment. 

The clusters use the IRIS [Dataset.](../..tests/testdata/iris_with_spaces_full.csv). 


## Calling this Model

You can test this mode with DRUM by running the following command: 
```
drum validation -cd . --target-type=multiclass --input=/Users/luke.shulman/Projects/datarobot-user-models/tests/testdata/iris_with_spaces_full.csv --class-labels-file class_labels.txt
```
or score with 

```
drum validation -cd . --target-type=multiclass --input=/Users/luke.shulman/Projects/datarobot-user-models/tests/testdata/iris_with_spaces_full.csv --class-labels-file class_labels.txt
```

