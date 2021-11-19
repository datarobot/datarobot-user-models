import pickle
from sklearn.tree import DecisionTreeRegressor


def transform(X, model):
    return X.drop("ISO_fr", axis="columns")


def fit(
    X, y, output_dir, class_order=None, row_weights=None, **kwargs,
):
    feature_columns = [c for c in X.columns if c != "some-weights"]
    assert feature_columns == [
        "TPA(ADJ)_fr",
        "AVG_fr",
        "OBP_fr",
        "SLG_fr",
        "OPS_fr",
        "HRPA_fr",
        "BBPA_fr",
        "SOPA_fr",
        "KW_fr",
        "XBH/AB_fr",
        "TPA(ADJ)_so",
        "AVG_so",
        "OBP_so",
        "SLG_so",
        "OPS_so",
        "ISO_so",
        "HRPA_so",
        "BBPA_so",
        "SOPA_so",
        "KW_so",
        "XBH/AB_so",
        "TPA(ADJ)_jr",
        "AVG_jr",
        "OBP_jr",
        "SLG_jr",
        "OPS_jr",
        "ISO_jr",
        "HRPA_jr",
        "BBPA_jr",
        "SOPA_jr",
        "KW_jr",
        "XBH/AB_jr",
    ]
    model = DecisionTreeRegressor()
    model.fit(X, y)
    with open("{}/model.pkl".format(output_dir), "wb") as f:
        pickle.dump(model, f)
