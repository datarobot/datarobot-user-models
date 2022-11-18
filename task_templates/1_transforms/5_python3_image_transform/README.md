# Image transform template

This transform example shows how to handle image data in a manner that is compatible
with DataRobot.  The example itself transforms an RGB image into a grayscale one and returns
it as a base64 encoded string.  There are helper methods in img_utils.py that handle conversion
to and from base 64 encoding.  

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir task_templates/1_transforms/5_python3_image_transform --input tests/testdata/cats_dogs_small_training.csv --target-type transform --target class`
If the command succeeds, your code is ready to be uploaded. 