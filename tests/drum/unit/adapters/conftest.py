import numpy as np
import pandas as pd
import pytest
import scipy.sparse


@pytest.fixture
def num_rows():
    return 10


@pytest.fixture
def num_cols():
    return 5


@pytest.fixture
def column_names(num_cols):
    return ["c" + str(i) for i in range(num_cols)]


@pytest.fixture
def target_col_name():
    return "target_col"


@pytest.fixture
def weights_col_name():
    return "weights_col_name"


@pytest.fixture
def col_data(num_rows):
    return np.array(range(num_rows))  # each col has values 0, 1, 2, ..., (num_rows - 1)


@pytest.fixture
def target_data_offset():
    return 0.1


@pytest.fixture
def target_series(col_data, target_data_offset, target_col_name):
    # offset target data so its not equal to X's col values
    return pd.Series(col_data + target_data_offset, name=target_col_name)


@pytest.fixture
def weights_data_offset():
    return 0.2


@pytest.fixture
def weights_series(col_data, weights_data_offset, weights_col_name):
    # offset weights data so its not equal to X's or target's col values
    return pd.Series(col_data + weights_data_offset, name=weights_col_name)


@pytest.fixture
def dense_df(col_data, num_cols, column_names):
    data = np.repeat([[x] for x in col_data], num_cols, axis=1)
    return pd.DataFrame(data, columns=column_names)


@pytest.fixture
def dense_df_with_target(dense_df, target_series):
    return pd.concat([dense_df, target_series], axis=1)


@pytest.fixture
def dense_df_with_target_and_weights(dense_df, target_series, weights_series):
    return pd.concat([dense_df, target_series, weights_series], axis=1)


@pytest.fixture
def sparse_df(dense_df, column_names):
    data = dense_df.values
    sparse_data = scipy.sparse.coo_matrix(data)

    sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_data)
    sparse_df.columns = column_names
    return sparse_df


@pytest.fixture
def dense_csv(dense_df, df_to_temporary_file):
    with df_to_temporary_file(dense_df) as filename:
        yield filename


@pytest.fixture
def dense_csv_with_target(dense_df_with_target, df_to_temporary_file):
    with df_to_temporary_file(dense_df_with_target) as filename:
        yield filename


@pytest.fixture
def dense_csv_with_target_and_weights(dense_df_with_target_and_weights, df_to_temporary_file):
    with df_to_temporary_file(dense_df_with_target_and_weights) as filename:
        yield filename


@pytest.fixture
def sparse_mtx(sparse_df, df_to_temporary_file):
    with df_to_temporary_file(sparse_df) as filename:
        yield filename


@pytest.fixture
def target_csv(target_series, df_to_temporary_file):
    with df_to_temporary_file(target_series) as filename:
        yield filename


@pytest.fixture
def weights_csv(weights_series, df_to_temporary_file):
    with df_to_temporary_file(weights_series) as filename:
        yield filename


@pytest.fixture
def sparse_column_names_file(column_names, df_to_temporary_file):
    with df_to_temporary_file(pd.DataFrame(column_names), header=False) as filename:
        yield filename
