from __future__ import absolute_import

import os
import pytest
import datarobot as dr

BASE_MODEL_TEMPLATES_DIR = "model_templates"


class TestInferenceModelTemplates(object):
    @pytest.mark.parametrize(
        "model_template, language, env, dataset, target_type, target, pos_label, neg_label, class_labels_file",
        [
            (
                "inference/java_codegen",
                "java",
                "java_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/h2o_pojo/binary",
                "java",
                "java_drop_in_env",
                "binary_testing_data",
                dr.TARGET_TYPE.BINARY,
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
                None,
            ),
            (
                "inference/h2o_mojo/binary",
                "java",
                "java_drop_in_env",
                "binary_testing_data",
                dr.TARGET_TYPE.BINARY,
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
                None,
            ),
            (
                "inference/h2o_pojo/regression",
                "java",
                "java_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_keras",
                "python",
                "keras_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_keras_joblib",
                "python",
                "keras_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_keras_vizai_joblib",
                "python",
                "keras_drop_in_env",
                "binary_vizai_testing_data",
                dr.TARGET_TYPE.BINARY,
                "class",
                "dogs",
                "cats",
                None,
            ),
            (
                "inference/python3_pytorch",
                "python",
                "pytorch_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_sklearn",
                "python",
                "sklearn_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_unstructured",
                "python",
                "sklearn_drop_in_env",
                # datafile here is only a stub, as unstructured model testing only performs start up check
                "regression_testing_data",
                dr.TARGET_TYPE.UNSTRUCTURED,
                None,
                None,
                None,
                None,
            ),
            (
                "inference/python3_xgboost",
                "python",
                "xgboost_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/r_lang",
                "r",
                "r_drop_in_env",
                "regression_testing_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
            (
                "inference/python3_pmml",
                "python",
                "pmml_drop_in_env",
                "binary_testing_data",
                dr.TARGET_TYPE.BINARY,
                "Species",
                "Iris-setosa",
                "Iris-versicolor",
                None,
            ),
            (
                "inference/python3_pytorch_multiclass",
                "python",
                "pytorch_drop_in_env",
                "multiclass_testing_data",
                dr.TARGET_TYPE.MULTICLASS,
                "class",
                None,
                None,
                "model_templates/inference/python3_pytorch_multiclass/class_labels.txt",
            ),
            (
                "inference/python3_sklearn",
                "python",
                "sklearn_drop_in_env",
                "regression_testing_arrow_data",
                dr.TARGET_TYPE.REGRESSION,
                "MEDV",
                None,
                None,
                None,
            ),
        ],
    )
    def test_inference_model_templates(
        self,
        request,
        model_template,
        language,
        env,
        dataset,
        target_type,
        target,
        pos_label,
        neg_label,
        class_labels_file,
    ):
        env_id, env_version_id = request.getfixturevalue(env)
        test_data_id = request.getfixturevalue(dataset)

        create_params = dict(
            name=model_template,
            target_type=target_type,
            description=model_template,
            language=language,
        )

        if target is not None:
            create_params.update({"target_name": target})

        if target_type == dr.TARGET_TYPE.BINARY:
            create_params.update(
                {"positive_class_label": pos_label, "negative_class_label": neg_label}
            )
        elif target_type == dr.TARGET_TYPE.MULTICLASS:
            create_params.update({"class_labels_file": class_labels_file})

        model = dr.CustomInferenceModel.create(**create_params)

        model_version = dr.CustomModelVersion.create_clean(
            custom_model_id=model.id,
            base_environment_id=env_id,
            folder_path=os.path.join(BASE_MODEL_TEMPLATES_DIR, model_template),
        )

        if model_version.dependencies:
            dr.CustomModelVersionDependencyBuild.start_build(model.id, model_version.id)

        test = dr.CustomModelTest.create(
            custom_model_id=model.id,
            custom_model_version_id=model_version.id,
            dataset_id=test_data_id,
        )

        assert test.overall_status == "succeeded"
