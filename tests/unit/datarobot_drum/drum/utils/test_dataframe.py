import pandas as pd
import pytest

from datarobot_drum.drum.utils.dataframe import extract_additional_columns


class TestExtractAdditionalColumns:
    @pytest.fixture(params=[lambda x: x, lambda x: list(reversed(x))])
    def class_ordering(self, request):
        return request.param

    def test_one_prediction_column_without_additional_columns(self):
        prediction_column = "Predictions"
        result_df = pd.DataFrame({prediction_column: [0.9, 0.8]})
        predictions, extra_model_output = extract_additional_columns(result_df, [prediction_column])
        assert predictions.equals(result_df)
        assert extra_model_output is None

    def test_multiple_prediction_columns_without_additional_columns(self, class_ordering):
        prediction_columns = class_ordering(["0", "1"])
        result_df = pd.DataFrame([[0.9, 0.1], [0.65, 0.35], [0.8, 0.2]], columns=prediction_columns)
        predictions, extra_model_output = extract_additional_columns(result_df, prediction_columns)
        assert predictions.equals(result_df)
        assert extra_model_output is None

    def test_one_prediction_column_with_one_additional_column(self):
        prediction_column = "Predictions"
        additional_column = "mean"
        result_df = pd.DataFrame(
            [[0.124, 0.3], [2.3, 1.2]], columns=[prediction_column, additional_column]
        )
        predictions, extra_model_output = extract_additional_columns(result_df, [prediction_column])
        assert predictions.equals(pd.DataFrame({prediction_column: [0.124, 2.3]}))
        assert extra_model_output.equals(pd.DataFrame({additional_column: [0.3, 1.2]}))

    def test_one_prediction_column_with_multiple_additional_columns(self, class_ordering):
        prediction_column = "Predictions"
        additional_columns = class_ordering(["A", "B", "C"])
        result_df = pd.DataFrame(
            [[2.3, "high", "fast", 55]], columns=[prediction_column] + additional_columns
        )
        predictions, extra_model_output = extract_additional_columns(result_df, [prediction_column])
        assert predictions.equals(pd.DataFrame({prediction_column: [2.3]}))
        assert extra_model_output.equals(
            pd.DataFrame([["high", "fast", 55]], columns=additional_columns)
        )

    def test_multiple_prediction_columns_with_one_additional_column(self, class_ordering):
        prediction_columns = class_ordering(["0", "1"])
        additional_column = "message"
        result_df = pd.DataFrame(
            [[0.2, 0.8, "Hello"]], columns=prediction_columns + [additional_column]
        )
        predictions, extra_model_output = extract_additional_columns(result_df, prediction_columns)
        assert predictions.equals(pd.DataFrame([[0.2, 0.8]], columns=prediction_columns))
        assert extra_model_output.equals(pd.DataFrame({additional_column: ["Hello"]}))

    def test_multiple_prediction_columns_with_multiple_additional_columns(self, class_ordering):
        prediction_columns = class_ordering(["0", "1"])
        additional_columns = class_ordering(["A", "B", "C"])
        result_df = pd.DataFrame(
            [[0.2, 0.8, "high", "fast", 55]], columns=prediction_columns + additional_columns
        )
        predictions, extra_model_output = extract_additional_columns(result_df, prediction_columns)
        assert predictions.equals(pd.DataFrame([[0.2, 0.8]], columns=prediction_columns))
        assert extra_model_output.equals(
            pd.DataFrame([["high", "fast", 55]], columns=additional_columns)
        )
