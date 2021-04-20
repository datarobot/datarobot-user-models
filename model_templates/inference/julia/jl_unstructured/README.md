
## Usage

```
drum score --code-dir model_templates/inference/julia/jl_unstructured --target-type unstructured --input tests/testdata/unstructured_data.txt --verbose
```

changing content type latin1

```
drum score --code-dir $CODE_DIR/jl_unstructured --target-type unstructured --input /opt/code/drum/tests/testdata/unstructured_data.txt --verbose --content-type "text/my_text;charset=latin1"
```

changing return to binary

```
drum score --code-dir $CODE_DIR/jl_unstructured --target-type unstructured --input /opt/code/drum/tests/testdata/unstructured_data.txt --verbose --query "ret_mode=binary"
```