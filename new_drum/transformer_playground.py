import sys
import pandas as pd
import numpy as np

# Make anything from new_drum importable
sys.path.insert(0, "./new_drum")

from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    def fit(self, X, y, **kwargs):

        # compute medians for all numeric features on training data, store them in a dictionary
        self.fit_data = X.median(axis=0, numeric_only=True, skipna=True).to_dict()

        # traing your mdoel
        # self.fit = my_model.fit()
        # self.my_saved_model_file = 'xyz.pt'
        # my_model_artifact = self.my_saved_model_file
        # my_saved_model = 'abc.h5'

        return self

    def save(self, artifact_directory):
        # self.my_saved_model_file = self.fit.dump(artifact_directory + self.my_saved_model_file)
        with open(artifact_directory + Serializable.default_filename, "wb") as fp:
            pickle.dump(self, fp)
        return self

    @classmethod
    def load(cls, artifact_directory):
        with open(artifact_directory + Serializable.default_filename, "rb") as fp:
            deserialized_object = pickle.load(fp)

        if not isinstance(deserialized_object, cls):
            raise ValueError("load method must return a {} class".format(cls.__name__))
        return deserialized_object

    def transform(self, X, **kwargs):

        # fillna can take either a value or a method
        return X.fillna(self.fit_data)


def generate_X_y(num_rows, num_cols):
    column_names = [str(i) for i in range(num_cols)]
    X = pd.DataFrame(np.random.randint(0, num_rows, size=(num_rows, num_cols)))
    X.columns = column_names

    y = pd.Series([0] * num_rows)
    return X, y


X, y = generate_X_y(100, 4)
transformer = CustomTask()
transformer.fit(X, y).save(artifact_directory=".")

transformer = CustomTask.load(".")
output = transformer.transform(X)

assert output.shape == (100, 4)
