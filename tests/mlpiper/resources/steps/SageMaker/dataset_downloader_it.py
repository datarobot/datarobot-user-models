import gzip
import pandas as pd
import pickle
import urllib.request
import time

from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components import ConnectableComponent
from mlpiper.common import os_util


class DatasetDownloaderIT(ConnectableComponent):
    DEFAULT_DATASET_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    logger = None
    last_update_time = time.time()

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)
        self._dataset_url = None
        self._train_set_local_csv_filepath = None
        self._valid_set_local_csv_filepath = None
        self._test_set_local_csv_filepath = None
        DatasetDownloaderIT.logger = self._logger

    def _materialize(self, parent_data_objs, user_data):

        self._init_params()

        tmp_dataset_filepath = os_util.tmp_filepath()
        self._logger.info(
            "Temporary dataset file path: {}".format(tmp_dataset_filepath)
        )

        try:
            urllib.request.urlretrieve(
                self._dataset_url,
                tmp_dataset_filepath,
                reporthook=DatasetDownloaderIT._download_report_hook,
            )
            self._logger.info("Dataset download completed ... 100%")

            train_set, valid_set, test_set = (None, None, None)
            with gzip.open(tmp_dataset_filepath, "rb") as f:
                loaded_artifacts = pickle.load(f, encoding="latin1")
                try:
                    train_set, valid_set, test_set = loaded_artifacts
                except ValueError:
                    try:
                        train_set, valid_set = loaded_artifacts
                    except ValueError:
                        train_set = loaded_artifacts

            self._logger.info(
                "Dataset downloaded and loaded! "
                + "#samples in train set: {}, ".format(
                    len(train_set[0]) if train_set else None
                )
                + "#samples in valid set: {}, ".format(
                    len(valid_set[0]) if valid_set else None
                )
                + "#samples in test set: {}".format(len(test_set[0]))
                if test_set
                else None
            )

            if self._train_set_local_csv_filepath and train_set:
                self._save_to_csv(train_set[0], self._train_set_local_csv_filepath)

            if self._valid_set_local_csv_filepath and valid_set:
                self._save_to_csv(valid_set[0], self._valid_set_local_csv_filepath)

            if self._test_set_local_csv_filepath and test_set:
                self._save_to_csv(test_set[0], self._test_set_local_csv_filepath)

            return [train_set, valid_set, test_set]
        except Exception as e:
            msg = "Failed to download and read dataset!\n{}".format(e)
            self._logger.error(msg)
            raise MLPiperException(msg)
        finally:
            self._logger.info(
                "Cleaning up temporary dataset file path: {}".format(
                    tmp_dataset_filepath
                )
            )
            os_util.remove_file_safely(tmp_dataset_filepath)

    def _init_params(self):
        self._dataset_url = self._params.get("dataset_url")
        if not self._dataset_url:
            self._dataset_url = DatasetDownloaderIT.DEFAULT_DATASET_URL
        self._logger.info("Dataset url to download is '{}'".format(self._dataset_url))

        self._train_set_local_csv_filepath = self._params.get(
            "train_set_local_csv_filepath"
        )
        self._valid_set_local_csv_filepath = self._params.get(
            "valid_set_local_csv_filepath"
        )
        self._test_set_local_csv_filepath = self._params.get(
            "test_set_local_csv_filepath"
        )

    @staticmethod
    def _download_report_hook(chunk_number, chunk_max_size, download_total_size):
        now = time.time()
        if now - DatasetDownloaderIT.last_update_time > 1.0:
            DatasetDownloaderIT.last_update_time = now
            percent = 100.0 * chunk_number * chunk_max_size / download_total_size
            DatasetDownloaderIT._download_report(percent)

    @staticmethod
    def _download_report(percent):
        DatasetDownloaderIT.logger.info(
            "Dataset download in progress ... {:.2f}%".format(percent)
        )

    def _save_to_csv(self, dataset, filepath):
        self._logger.info("Saving to csv ... {}".format(filepath))
        pd.DataFrame(dataset).to_csv(filepath, header=None, index=False)
