## Java Unstructured Inference Model Template

This model is intended to work with the [Java Drop-In Environment](../../public_dropin_environments/java_codegen/).

The data is just a text file: [unstructured_data.txt](../../tests/testdata/unstructured_data.txt).

The requires the model to implement the `BasePredictor` interface.  

## Instructions

This requires an install of predictors.jar at [custom_model_runner/datarobot_drum/drum/language_predictors/java_predictor](custom_model_runner/datarobot_drum/drum/language_predictors/java_predictor) built and install with `mvn package install`. 

There are two required environment variables: 

`export DRUM_JAVA_CUSTOM_PREDICTOR_CLASS=custom.CustomModel`

`export DRUM_JAVA_CUSTOM_CLASS_PATH=/full/path/to/model_templates/java_unstructured/model/custom-model-0.1.0.jar`


### To run locally using 'drum'

File paths are relative to `./datarobot-user-models`:   

`drum score --code-dir model_templates/java_unstructured/model --target-type unstructured --input tests/testdata/unstructured_data.txt`

#### Running with additional params

The following example shows values passed can be handled by the unstructured java model.  

DRUM will convert all data to a bytes array before passing it to the java model.  Morevoer, mimetype, charset, and query will be parsed from the provided arguments, and will be passed to the java model as a String, String, and Map<String, String> respectively.  

As far as the return from your java model, you can return a string (plain text or json string), bytes or a bytes array.  

The custom java model will need to implement function with the following signature: `public <T> T predict_unstructured(byte[] inputBytes, String mimetype, String charset, Map<String, String> query)`, using a switch in the function will allow you to return one of the afformentioned data types.  

##### Example 1
Command:   
`drum score --code-dir model_templates/java_unstructured/model --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose`

Output:
```
incoming mimetype: text/plain
Incoming Charset: utf8
Incoming Query: {}
Incoming data: rough road leads to the stars 
через тернии к звездам
```
The `Model` value represents what is returned by the `load_model` hook.    
In this example, the content type was not specified so the default values are passed: `mimetype=text/plain` and `charset=utf8`. In this case, data is treated as text and passed as`str` into the hook.  
The query params are empty because the `--query` argument (optional) was not provided.

##### Example 2
The following example demonstrates a custom *textual* content type.  
DRUM handles `mimetype` starting with `text/` or `application/json` as a textual content type.  
Command:   

`drum score --code-dir model_templates/java_unstructured/model --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose --content-type "text/my_text;charset=windows-1252"`

Output:
```
incoming mimetype: text/my_text
Incoming Charset: windows-1252
Incoming Query: {}
Incoming data: rough road leads to the stars 
Ñ‡ÐµÑ€ÐµÐ· Ñ‚ÐµÑ€Ð½Ð¸Ð¸ Ðº Ð·Ð²ÐµÐ·Ð´Ð°Ð¼
```

In this example, the result is mostly the same; the content type params are `mimetype=text/my_text` and `charset=windows-1252`.  
The text is unreadable because it has been decoded using the `windows-1252` charset, while the input text file is encoded with `utf8` which is the default for Linux systems.  
> Note that in the most cases you are working with the text data with default encoding, or binary data. Otherwise you are responsible for properly handling encoding or decoding.


##### Example 3
This example provides a custom *binary* content type and query params.  
DRUM handles any data with `mimetype` that does not start with `text/` or `application/json` as binary.  
Command:   
`drum score --code-dir model_templates/java_unstructured/model --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose --content-type "application/octet-stream" --query "ret_mode=text" --output out_file`

Output:
```
Model:  dummy
incoming mimetype: application/octet-stream
Incoming Charset: utf-8
Incoming Query: {"ret_mode":"text"}
Incoming data: rough road leads to the stars 
через тернии к звездам
```

The values are `mimetype=application/stream` and `charset` is missing as it was not passed, because not required for binary data.   Since `charset` is missing, java unstructured models default to `utf-8`. 

> Note: If you don't want to pass the textual mimetype so that DRUM can decode data,
> you still can pass mimetype for binary data and charset and decode the data yourself, e.g. `application/octet-stream;charset=<your_charset>`.
> This depends on the particular task. 

The data being passed to the unstructured java model will always be a byte array.  
The data is treated as binary, so the type is `bytes`.
The incoming query params are `ret_mode=text` or `ret_mode=binary` to output data as either str or bytes.

Open output file `out_file`. It should contain text `10`, which is the word count for the data file.  

Now change `ret_mode` to `binary`. You will be unable to open `out_file` as a text file. On a Linux system, use the command `xxd out_file`. You should see `00000000: 0000 000a`.   
`0000 000a` - is 4 bytes, representing the value `10`.
