"""
Copyright 2026 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from tests.unit.datarobot_drum.drum.helpers import loaded_proprietary_modules


def test_drum_server_modules_do_not_import_proprietary_libraries(tmp_path):
    script = f"""
    import datarobot_drum.drum.drum
    import datarobot_drum.drum.root_predictors.prediction_server
    from datarobot_drum.drum.language_predictors.base_language_predictor import (
        BaseLanguagePredictor,
    )

    class Predictor(BaseLanguagePredictor):
        def _predict(self, **kwargs):
            pass

        def _transform(self, **kwargs):
            pass

        def has_read_input_data_hook(self):
            return False

    Predictor().configure(
        {{
            "target_type": "regression",
            "__custom_model_path__": {str(tmp_path)!r},
            "monitor": False,
            "allow_dr_api_access": False,
        }}
    )
    """
    assert loaded_proprietary_modules(script) == []
