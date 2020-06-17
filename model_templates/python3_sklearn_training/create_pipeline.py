from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

# Feel free to delete which ever one of these you are not using. Having these functions in a
# separate file was mostly done so that the user could see how to split up python code into multiple
# files


def make_classifier(X):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    classifier = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", SVC(probability=True))]
    )

    return classifier


def make_regressor(X):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    regressor = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", SVR())])
    return regressor
