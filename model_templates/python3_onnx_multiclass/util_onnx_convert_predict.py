import torch.onnx

ONNX_MODEL_PATH = "multiclass_SDSS.onnx"


def convert_to_onnx(model):
    model.eval()
    dummy_input = torch.zeros(5, 10)
    dynamic_axes = {'modelInput': {0: 'batch'}, 'modelOutput': {0: 'batch'}}
    torch.onnx.export(model,
                      args=dummy_input,
                      f=ONNX_MODEL_PATH,
                      export_params=True,
                      opset_version=10,
                      input_names=['modelInput'],
                      output_names=['modelOutput'],
                      dynamic_axes=dynamic_axes)
    print('model converted to ONNX')


def load_and_convert():
    model = torch.load("artifact.pth")
    model.eval()
    print(model)
    convert_to_onnx(model)


if __name__ == "__main__":
    load_and_convert()

    import onnx
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    check_result = onnx.checker.check_model(onnx_model)
    print(check_result)

    import onnxruntime
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    import pandas as pd
    import numpy as np
    import pickle

    test_input = "/Users/asli.demiroz/repos/datarobot-user-models/tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv"
    test_df = pd.read_csv(test_input)

    with open("preprocessor.pkl", mode="rb") as f:
        preprocessor = pickle.load(f)
    if preprocessor is None:
        raise ValueError("Preprocessor not loaded")

    transformed_df = pd.DataFrame(preprocessor.transform(test_df))

    ort_inputs = {ort_session.get_inputs()[0].name: transformed_df.to_numpy(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)