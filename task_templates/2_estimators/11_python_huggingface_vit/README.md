# Huggingface Demo Template - VIT
This template shows the use of the Huggingface library, specifically for a vision classification task using
ViT.  

## Setup
Download the base pretrained model from huggingface and place it in this directory.  It is not saved here already
due to file size.  Custom tasks run within DataRobot do not have access to the internet, necessitating adding the
pretrained model files to the task and uploading it.  
`git clone https://huggingface.co/google/vit-base-patch16-224`

`drum fit --code-dir task_templates/2_estimators/11_python_huggingface_vit --input tests/testdata/cats_dogs_small_training.csv --target-type binary --target class`
