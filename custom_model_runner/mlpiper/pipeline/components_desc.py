import fnmatch
import io
import json
import pkg_resources
import os
import re
import sys

from mlpiper.common.base import Base
import mlpiper.common.constants as MLPiperConstants
from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.pipeline import json_fields


class ComponentsDesc(Base):
    CODE_COMPONETS_MODULE_NAME = "parallelm.code_components"
    COMPONENT_JSON_FILE = "component.json"
    COMPONENT_METADATA_REF_FILE = "__component_reference__.json"

    def __init__(self, ml_engine=None, pipeline=None, comp_root_path=None, args=None):
        super(ComponentsDesc, self).__init__(
            ml_engine.get_engine_logger(self.logger_name()) if ml_engine else None
        )
        self._ml_engine = ml_engine
        self._pipeline = pipeline
        self._comp_root_path = comp_root_path
        self._args = args
        self._comp_scanner = None

    @staticmethod
    def handle(args):
        ComponentsDesc(args).write_details()

    @staticmethod
    def next_comp_desc(root_dir):
        for root, _, files in os.walk(root_dir):
            for filename in files:
                comp_desc = ComponentsDesc._load_comp_desc(root, filename)
                if comp_desc:
                    yield root, comp_desc, filename

    @staticmethod
    def _load_comp_desc(root, filename):
        if filename.endswith(".json"):
            comp_json = os.path.join(root, filename)
            with io.open(comp_json, encoding="utf-8") as f:
                try:
                    comp_desc = json.load(f)
                except ValueError as ex:
                    raise MLPiperException(
                        "Found json file with invalid json format! "
                        "filename: {}, exception: {}".format(comp_json, str(ex))
                    )

            if ComponentsDesc.is_valid(comp_desc):
                # In the context of adding support for multiple engine types for a given
                # step, even if only a single engine is defined then turn it to a list
                engine_type = comp_desc[json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD]
                if not isinstance(engine_type, list):
                    comp_desc[json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD] = [engine_type]

                return comp_desc

        return None

    @staticmethod
    def is_valid(comp_desc):
        comp_desc_signature = [
            json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD,
            json_fields.COMPONENT_DESC_NAME_FIELD,
            json_fields.COMPONENT_DESC_LANGUAGE_FIELD,
            json_fields.COMPONENT_DESC_PROGRAM_FIELD,
        ]

        if set(comp_desc_signature) <= set(comp_desc):
            return True
        return False

    def write_details(self):
        out_file_path = (
            self._args.comp_desc_out_path
            if self._args.comp_desc_out_path
            else "./components_description.json"
        )

        components_desc = self.load(extended=False)

        with io.open(out_file_path, mode="w", encoding="utf-8") as f:
            json.dump(components_desc, f, indent=4)
            print("Components details were written successfully to: " + out_file_path)

    def _add_default_values(self, comp_desc):
        if json_fields.COMPONENT_DESC_USER_STAND_ALONE not in comp_desc:
            comp_desc[json_fields.COMPONENT_DESC_USER_STAND_ALONE] = True

    def load(self, extended=True):
        components_desc = []

        if not self._comp_root_path:
            try:
                # The following call to 'pkg_resources.resource_filename' actually extract
                # all the files from the component's egg file from
                # 'parallelm.code_components' folder
                self._comp_root_path = pkg_resources.resource_filename(
                    ComponentsDesc.CODE_COMPONETS_MODULE_NAME, ""
                )
                self._logger.info("Cached components are at: {}".format(self._comp_root_path))
            except ImportError:
                msg = "Either component's root path or component's egg file are missing!"
                self._logger.error(msg)
                raise MLPiperException(msg)

        comp_repo_info = ComponentScanner(self._comp_root_path, self._ml_engine).comp_repo_info

        engine_type = self._pipeline[json_fields.PIPELINE_ENGINE_TYPE_FIELD]
        for comp_type in self._get_next_comp_type_in_pipeline():
            if comp_type not in comp_repo_info[engine_type]:
                raise MLPiperException(
                    "Component '{}' is not registered under engine '{}'!"
                    " Please make sure the component is located under the configured"
                    " root path: {}".format(comp_type, engine_type, self._comp_root_path)
                )
            self._logger.info(
                "Handling step ... engine: {}, comp-type: {}".format(engine_type, comp_type)
            )
            comp_path = comp_repo_info[engine_type][comp_type]["root"]
            if comp_path not in sys.path:
                sys.path.insert(0, comp_path)

            comp_desc = comp_repo_info[engine_type][comp_type]["comp_desc"]
            if extended:
                comp_desc[json_fields.COMPONENT_DESC_ROOT_PATH_FIELD] = comp_path

            self._add_default_values(comp_desc)
            self._logger.debug("Component loaded: " + str(comp_desc))
            components_desc.append(comp_desc)

        return components_desc

    def _get_next_comp_type_in_pipeline(self):
        comp_already_handled = []
        for pipe_comp in self._pipeline[json_fields.PIPELINE_PIPE_FIELD]:
            comp_type = pipe_comp[json_fields.PIPELINE_COMP_TYPE_FIELD]
            if comp_type in comp_already_handled:
                continue
            comp_already_handled.append(comp_type)
            yield comp_type

    def _component_module_path(self, comp_type):
        comp_module = comp_type.rsplit(".", 1)[0]
        return os.path.join(self._comp_root_path, comp_module)

    def read_desc_file(self, comp_path):
        self._logger.debug("Reading component's metadata: {}".format(comp_path))
        comp_ref_json = os.path.join(comp_path, ComponentsDesc.COMPONENT_METADATA_REF_FILE)
        if os.path.isfile(comp_ref_json):
            with open(comp_ref_json, "r") as f:
                try:
                    comp_ref = json.load(f)
                except ValueError:
                    msg = "Failed to load (parse) component metadata's reference file! "
                    "ref-file: {}".format(comp_ref_json)
                    self._logger.error(msg)
                    raise MLPiperException(msg)

            metadata_filename = comp_ref[json_fields.COMPONENT_METADATA_REF_FILE_NAME_FIELD]
            comp_desc = ComponentsDesc._load_comp_desc(comp_path, metadata_filename)
        else:
            # Try to find any valid component's description file
            comp_desc_gen = ComponentsDesc.next_comp_desc(comp_path)
            try:
                # next() is called only once, because only one component JSON file is expected.
                _, comp_desc, _ = next(comp_desc_gen)
            except StopIteration:
                comp_desc = None

        if not comp_desc:
            msg = "Failed to find any valid component's json desc! comp_path: {}".format(comp_path)
            self._logger.error(msg)
            raise MLPiperException(msg)

        return comp_desc


class ComponentScanner(Base):
    def __init__(self, root_dir, ml_engine=None):
        super(ComponentScanner, self).__init__(
            ml_engine.get_engine_logger(self.logger_name()) if ml_engine else None
        )
        self._root_dir = root_dir
        self._comp_repo_info = None

    @property
    def comp_repo_info(self):
        if not self._comp_repo_info:
            self._comp_repo_info = self._scan_dir()
        return self._comp_repo_info

    def _scan_dir(self):
        """
        Scanning a directory returning a map of components:
        {
            "name1": {
                "directory": path_relative_to_root_dir
            }
        }
        :return:
        """
        comps = {}
        self._logger.debug("Scanning {}".format(self._root_dir))
        for root, comp_desc, comp_filename in ComponentsDesc.next_comp_desc(self._root_dir):
            for engine_type in comp_desc[json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD]:
                comps.setdefault(engine_type, {})

                comp_name = comp_desc[json_fields.COMPONENT_DESC_NAME_FIELD]

                if self._filter_component(comp_name, engine_type, root, comp_filename, comps):
                    continue

                comps[engine_type][comp_name] = {}
                comps[engine_type][comp_name]["comp_desc"] = comp_desc
                comps[engine_type][comp_name]["root"] = root
                comps[engine_type][comp_name]["files"] = self._include_files(root, comp_desc)
                # Always include current component json file regardless of its name.
                comps[engine_type][comp_name]["files"].append(comp_filename)
                comps[engine_type][comp_name]["comp_filename"] = comp_filename

                self._logger.debug(
                    "Found component, root: {}, engine: {}, name: {}".format(
                        root, engine_type, comp_name
                    )
                )
        return comps

    def _filter_component(self, comp_name, engine_type, root, comp_filename, comps):
        """
        If a component with the same name/type was already loaded for the given engine
        then check the last modified time for both components. The latest modified
        component will be the one that will be registered under the given engine.
        """
        if comp_name in comps[engine_type]:
            already_loaded_comp = comps[engine_type][comp_name]

            current_step_path = os.path.join(root, comp_filename)
            already_loaded_step_path = os.path.join(
                already_loaded_comp["root"], already_loaded_comp["comp_filename"]
            )

            current_step_last_mtime = os.path.getmtime(current_step_path)
            already_loaded_step_mtime = os.path.getmtime(already_loaded_step_path)

            if current_step_last_mtime < already_loaded_step_mtime:
                self._logger.info(
                    "Component already loaded! Skipping! engine: {}, comp: {}, "
                    "prev_comp_path: {}, new_comp_path: {}".format(
                        engine_type,
                        comp_name,
                        already_loaded_step_path,
                        current_step_path,
                    )
                )
                return True

            self._logger.info(
                "Overriding an already loaded component! engine: {}, comp: {}, "
                "prev_comp_path: {}, new_comp_path: {}".format(
                    engine_type, comp_name, already_loaded_step_path, current_step_path
                )
            )

        return False

    def _include_files(self, comp_root, comp_desc):
        include_patterns = self._parse_patterns(
            comp_desc.get(json_fields.COMPONENT_DESC_INCLUDE_GLOB_PATTERNS)
        )
        exclude_patterns = self._parse_patterns(
            comp_desc.get(json_fields.COMPONENT_DESC_EXCLUDE_GLOB_PATTERNS)
        )

        # Add "requirements.txt" if includeGlobPattern is defined,
        # so requirements file will be always copied.
        if len(include_patterns):
            if os.path.exists(os.path.join(comp_root, MLPiperConstants.REQUIREMENTS_FILENAME)):
                include_patterns.append(MLPiperConstants.REQUIREMENTS_FILENAME)

        included_files = []
        for root, _, files in os.walk(comp_root):
            for f in files:
                rltv_path = os.path.relpath(root, comp_root)
                filepath = os.path.join(rltv_path, f) if rltv_path != "." else f
                if self._path_included(filepath, include_patterns, exclude_patterns):
                    # There can be several comp JSONs in one folder.
                    # Don't include any of them, even related to current component,
                    # it will be included automatically
                    if ComponentsDesc._load_comp_desc(comp_root, f):
                        continue

                    included_files.append(filepath)

        return included_files

    def _parse_patterns(self, pattern):
        if not pattern:
            return []
        return re.sub("\\s+", "", pattern.strip()).split("|")

    def _path_included(self, file, include_patterns, exclude_patterns):
        # For any given path, assume first that it should be included. This is the default
        # if no 'include' matcher exists. If 'include' matcher exists, assume that the path
        # should be excluded, then check the inclusion condition and set it accordingly
        included = False if include_patterns else True

        for pattern in include_patterns:
            if fnmatch.fnmatch(file, pattern):
                included = True
                break

            # For a any given path, only if it is supposed to be included,
            # check for exclusion condition.
        if included:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file, pattern):
                    included = False
                    break

        return included
