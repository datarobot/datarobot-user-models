from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle

ONNX_MODEL_PATH = "anomaly_juniors.onnx"


def convert_to_onnx(model):
    initial_type = [('float_input', FloatTensorType([None, 34]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onx.SerializeToString())
    print('model converted to ONNX')


def load_and_convert():
    model = None
    with open('artifact.pkl', "rb") as picklefile:
        try:
            model = pickle.load(picklefile, encoding="latin1")
        except TypeError:
            model = pickle.load(picklefile)
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

    test_input = "/Users/asli.demiroz/repos/datarobot-user-models/tests/testdata/juniors_3_year_stats_regression.csv"
    test_df = pd.read_csv(test_input)
    print(test_df.shape)

    test_df = test_df.fillna(-99999)

    ort_inputs = {ort_session.get_inputs()[0].name: test_df.to_numpy(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
