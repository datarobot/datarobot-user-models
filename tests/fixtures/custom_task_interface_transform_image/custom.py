"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from typing import Union
import pandas as pd

from img_utils import (
    b64_to_img,
    img_to_b64,
    img_to_grayscale,
)
from datarobot_drum.custom_task_interfaces import TransformerInterface


class CustomTask(TransformerInterface):
    @staticmethod
    def _process_image(raw_data: Union[str, bytes]) -> bytes:
        img = b64_to_img(raw_data)
        img = img_to_grayscale(img)
        return img_to_b64(img)

    def fit(self, X, y, **kwargs):
        """ The transform hook below is stateless and doesn't require a trained transformer, so the fit is blank"""
        pass

    def transform(self, X, **kwargs):
        """Transform function that does not require any fit hook above"""

        return X.applymap(self._process_image)
