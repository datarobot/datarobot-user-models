# -*- coding: utf-8 -*-
"""

Copyright 2017 xuannianz github user

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THE FOLLOWING IS THE COPYRIGHT OF THE ORIGINAL DOCUMENT:
https://github.com/xuannianz/EfficientDet/blob/master/tfkeras.py
"""
from object_detection_utils import inject_tfkeras_modules, init_tfkeras_custom_objects
import efficientnet as model

EfficientNetB0 = inject_tfkeras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(model.EfficientNetB7)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
