## Python Unstructured Inference Model Template

This model is intended to work with the [Python 3 Scikit-Learn Drop-In Environment](../../public_dropin_environments/python3_sklearn/).
or any other python environment as it doesn't require any specific packages.

The data is just a text file: [unstructured_data.txt](../../tests/testdata/unstructured_data.txt).

The model doesn't use any artifacts, only `custom.py` which implements the `score_unstructured` method and performs a word count on the input data.

## Instructions
Create a new custom model with these files and use it locally in a virtual environment or with the Python Drop-In Environment.

### To run locally using 'drum'
File paths are relative to `./datarobot-user-models`:   
`drum score --code-dir model_templates/python3_unstructured --target-type unstructured --input tests/testdata/unstructured_data.txt`

#### Running with additional params
The following example shows how to control the types of the values passed into the `score_unstructured` hook.

Consider the following signature: `score_unstructured(model, data, query, **kwargs)`

##### Example 1
Command:   
`drum score --code-dir model_templates/python3_unstructured --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose`

Output:
```
Model:  dummy
Incoming content type params:  {'mimetype': 'text/plain', 'charset': 'utf8'}
Incoming data type:  <class 'str'>
Incoming data:  rough road leads to the stars 
через тернии к звездам

Incoming query params:  {}
```
The `Model` value represents what is returned by the `load_model` hook.    
In this example, the content type was not specified so the default values are passed: `mimetype=text/plain` and `charset=utf8`. In this case, data is treated as text and passed as`str` into the hook.  
The query params are empty because the `--query` argument (optional) was not provided.

##### Example 2
The following example demonstrates a custom *textual* content type.  
DRUM handles `mimetype` starting with `text/` or `application/json` as a textual content type.  
Command:   
`drum score --code-dir model_templates/python3_unstructured --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose --content-type "text/my_text;charset=latin1"`

Output:
```
Model:  dummy
Incoming content type params:  {'mimetype': 'text/my_text', 'charset': 'latin1'}
Incoming data type:  <class 'str'>
Incoming data:  rough road leads to the stars 
ÑÐµÑÐµÐ· ÑÐµÑÐ½Ð¸Ð¸ Ðº Ð·Ð²ÐµÐ·Ð´Ð°Ð¼

Incoming query params:  {}
```

In this example, the result is mostly the same; the content type params are `mimetype=text/my_text` and `charset=latin1`.  
The text is unreadable because it has been decoded using the `latin1` charset, while the input text file is encoded with `utf8` which is the default for Linux systems.  
> Note that in the most cases you are working with the text data with default encoding, or binary data. Otherwise you are responsible for properly handling encoding or decoding.


##### Example 3
This example provides a custom *binary* content type and query params.  
DRUM handles any data with `mimetype` that does not start with `text/` or `application/json` as binary.  
Command:   
`drum score --code-dir model_templates/python3_unstructured --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose --content-type "application/octet-stream" --query "ret_mode=text" --output out_file`

Output:
```
Model:  dummy
Incoming content type params:  {'mimetype': 'application/octet-stream'}
Incoming data type:  <class 'bytes'>
Incoming data:  b'rough road leads to the stars \n\xd1\x87\xd0\xb5\xd1\x80\xd0\xb5\xd0\xb7 \xd1\x82\xd0\xb5\xd1\x80\xd0\xbd\xd0\xb8\xd0\xb8 \xd0\xba \xd0\xb7\xd0\xb2\xd0\xb5\xd0\xb7\xd0\xb4\xd0\xb0\xd0\xbc\n'
Incoming query params:  {'ret_mode': 'text'}

```

The values are `mimetype=application/stream` and `charset` is missing as it was not passed, because not required for binary data.   
> Note: If you don't want to pass the textual mimetype so that DRUM can decode data,
> you still can pass mimetype for binary data and charset and decode the data yourself, e.g. `application/octet-stream;charset=<your_charset>`.
> This depends on the particular task. 

The data is treated as binary, so the type is `bytes`.
The incoming query params are `ret_mode=text`, used in `custom.py` to output data as either str or bytes.  
Open output file `out_file`. It should contain text `10`, which is the word count for the data file.  

Now change `ret_mode` to `binary`. You will be unable to open `out_file` as a text file. On a Linux system, use the command `xxd out_file`. You should see `00000000: 0000 000a`.   
`0000 000a` - is 4 bytes, representing integer `10`.
