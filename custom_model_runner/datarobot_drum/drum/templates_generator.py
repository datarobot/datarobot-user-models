"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import shutil
import logging
import time
import pprint
from jinja2 import Environment, FileSystemLoader

from datarobot_drum.drum.enum import (
    LOGGER_NAME_PREFIX,
    CUSTOM_FILE_NAME,
    CustomHooks,
    ArgumentsOptions,
    RunLanguage,
    TemplateType,
)

MODEL_TEMPLATE_README = "MODEL_README_TEMPLATE.md.j2"
MODEL_TEMPLATE_CUSTOM_PYTHON = "custom_python_template.py.j2"
MODEL_TEMPLATE_CUSTOM_R = "custom_r_template.R.j2"
MODEL_TEMPLATE_CUSTOM_JULIA = "custom_julia_template.jl.j2"


class CMTemplateGenerator:
    def __init__(self, template_type, language, dir):
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)

        self._template_type = template_type
        self._language = language
        self._dir = dir
        self._templates_dir = os.path.join(os.path.dirname(__file__), "../resource/templates")
        self._logger.debug("templates_dir: {}".format(self._templates_dir))
        self._file_loader = FileSystemLoader(self._templates_dir)

        self._language_info = {
            RunLanguage.PYTHON: {
                "model_template": MODEL_TEMPLATE_CUSTOM_PYTHON,
                "line_comment": "#",
                "suffix": ".py",
            },
            RunLanguage.R: {
                "model_template": MODEL_TEMPLATE_CUSTOM_R,
                "line_comment": "#",
                "suffix": ".R",
            },
            RunLanguage.JULIA: {
                "model_template": MODEL_TEMPLATE_CUSTOM_JULIA,
                "line_comment": "#",
                "suffix": ".jl",
            },
        }

    def generate(self):
        if self._template_type == TemplateType.MODEL:
            self._logger.debug("lang: {}".format(self._language))
            if self._language in [RunLanguage.PYTHON, RunLanguage.R, RunLanguage.JULIA]:
                self._generate_model_template()
            else:
                raise NotImplementedError(
                    "Language: {} is not supported yet".format(self._language)
                )
        elif self._template_type == TemplateType.ENV:
            raise NotImplementedError("Environment templates are not done yet .., soon")
        else:
            raise NotImplementedError(
                "template type: {} is not implemented".format(self._template_type)
            )

    def _copy_and_render(self, src, dst, token_values, prefix=None):
        self._logger.debug("vars: {}".format(pprint.pformat(token_values)))
        env = Environment(loader=self._file_loader)
        template = env.get_template(os.path.basename(src))
        output = template.render(**token_values)

        if prefix:
            output = "\n{}".format(prefix).join(("\n" + output.lstrip()).splitlines()).lstrip()

        with open(dst, "w") as dst_file:
            dst_file.write(output)

    def _generate_model_template(self):
        os.makedirs(self._dir)
        lang_info = self._language_info[self._language]
        custom_file_name = CUSTOM_FILE_NAME + lang_info["suffix"]

        variables_to_replace = {
            "gen_command": "{} new model --language {}".format(
                ArgumentsOptions.MAIN_COMMAND, self._language.value
            ),
            "gen_date": time.ctime(time.time()),
            "custom_name": custom_file_name,
        }

        self._logger.debug("Templates are at: {}".format(self._templates_dir))
        readme_src = os.path.join(self._templates_dir, MODEL_TEMPLATE_README)
        readme_dst = os.path.join(self._dir, "README.md")
        self._copy_and_render(readme_src, readme_dst, variables_to_replace)

        custom_src = os.path.join(self._templates_dir, lang_info["model_template"])
        custom_dst = os.path.join(self._dir, custom_file_name)
        self._logger.debug("src: {} dst:{}".format(custom_src, custom_dst))
        self._copy_and_render(
            custom_src, custom_dst, variables_to_replace, prefix=lang_info["line_comment"]
        )
