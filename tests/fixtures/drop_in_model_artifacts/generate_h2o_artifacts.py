## would require a pip install of h2o==3.34.0.1
import h2o
import logging
from h2o.estimators import H2OXGBoostEstimator, H2ORandomForestEstimator

assert h2o.__version__ == "3.34.0.1"


class GenerateH2OArtifacts(object):
    def __init__(self):
        logging.basicConfig(
            format="{} - %(levelname)s - %(asctime)s - %(message)s".format(self.__module__)
        )
        self.logger = logging.getLogger()
        self.logger.setLevel("INFO")

    def train(
        self,
        data_location,
        target,
        target_type,
        estimator=H2ORandomForestEstimator(ntrees=10, max_depth=5, min_rows=10),
    ):
        df = h2o.import_file(data_location)
        predictors = [c for c in df.columns if c not in [target, "Id", "objid"]]
        self.logger.info(f"training data is {data_location}")
        self.logger.info(f"target type is {target_type}")
        if target_type == "classification":
            self.logger.info(f"casting target {target} as factor for h2o")
            df[target] = df[target].asfactor()
        elif target_type == "regression":
            pass
        else:
            raise Exception(f"target type {target_type} not supported")
        # Split the dataset into a train and valid set:
        train, valid = df.split_frame(ratios=[0.8], seed=1234)
        # Build and train the model:
        self.logger.info("training estimator")
        estimator = estimator.train(
            x=predictors, y=target, training_frame=train, validation_frame=valid
        )
        self.logger.info("training complete")
        self.model = estimator
        # Eval performance:
        perf = estimator.model_performance()
        perf.show()

    def export(self, export_as="mojo", export_location="."):
        if export_as == "mojo":
            self.logger.info("exporting mojo to {export_location}")
            self.model.download_mojo(path=export_location, get_genmodel_jar=False, genmodel_name="")
        elif export_as == "pojo":
            self.logger.info(f"exporting pojo to {export_location}")
            self.logger.info("do NOT change the name of the java file")
            self.model.download_pojo(path=export_location, get_genmodel_jar=False, genmodel_name="")
        else:
            raise Exception(f"export as {export_as} : should be on of mojo or pojo")


if __name__ == "__main__":

    h2o.init()
    h2o_reg_artifacts = GenerateH2OArtifacts()
    data_location = "../../testdata/juniors_3_year_stats_regression.csv"
    target = "Grade 2014"
    target_type = "regression"
    h2o_reg_artifacts.train(data_location, target, target_type)
    h2o_reg_artifacts.export("mojo", export_location="../../../model_templates/h2o_mojo/regression")
    h2o_reg_artifacts.export("pojo", export_location="../../../model_templates/h2o_pojo/regression")
    h2o_reg_artifacts.export("mojo")
    h2o_reg_artifacts.export("pojo")

    # h2o_binary_artifacts= GenerateH2OArtifacts()
    # data_location = "../../testdata/iris_binary_training.csv"
    # target = "Species"
    # target_type = "classification"
    # h2o_binary_artifacts.train(data_location, target, target_type, estimator = H2OXGBoostEstimator())
    # h2o_binary_artifacts.export("mojo", export_location = "../../../model_templates/h2o_mojo/binary")
    # h2o_binary_artifacts.export("pojo", export_location = "../../../model_templates/h2o_pojo/binary")
    # h2o_binary_artifacts.export("mojo")
    # h2o_binary_artifacts.export("pojo")

    # h2o_multiclass_artifacts = GenerateH2OArtifacts()
    # data_location = "../../testdata/skyserver_sql2_27_2018_6_51_39_pm.csv"
    # target = "class"
    # target_type = "classification"
    # h2o_multiclass_artifacts.train(data_location, target, target_type)
    # h2o_multiclass_artifacts.export("mojo", export_location = "../../../model_templates/h2o_mojo/multiclass")
    # h2o_multiclass_artifacts.export("pojo", export_location = "../../../model_templates/h2o_pojo/multiclass")
    # h2o_multiclass_artifacts.export("mojo")
    # h2o_multiclass_artifacts.export("pojo")

    h2o.cluster().shutdown()
