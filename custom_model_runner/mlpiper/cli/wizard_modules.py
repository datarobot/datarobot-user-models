import os
from mlpiper.pipeline.component_language import ComponentLanguage
from mlpiper.pipeline.component_group import ComponentGroup
from mlpiper.pipeline.data_type import EngineType
from mlpiper.pipeline.component_model_behavior_type import ComponentModelBehaviorType

from mlpiper.pipeline import json_fields

from pypsi import wizard as wiz
from pypsi.ansi import AnsiCodes

import six


# getting __dict__ attributes and filtering only what is needed
language_options = [
    v for k, v in vars(ComponentLanguage).items() if not k.startswith("__")
]
engine_options = [v for k, v in vars(EngineType).items() if not k.startswith("__")]
group_options = [v for k, v in vars(ComponentGroup).items() if not k.startswith("__")]
model_behavior_options = [
    v for k, v in vars(ComponentModelBehaviorType).items() if not k.startswith("__")
]
boolean_options = ["True", "False"]
argument_type_options = [
    "string",
    "int",
    "long",
    "float",
    "double",
    "boolean",
    "sequence.string",
    "sequence.string",
]
argument_tag_options = ["model_dir", "input_model_path", None]


def options_to_str(options_list):
    """
    Helper method to create a sting out of a list of choice options.
    """
    tmp_list = ["{} - {}".format(i + 1, o) for i, o in enumerate(options_list)]
    return "\n".join(tmp_list)


def choice_validator(choices):
    """
    validator help to get value from options list.
    value can be a number in string representation or string itself(when default value is taken)
    """

    def validator(ns, value):

        if value is None:
            raise ValueError(
                "Invalid input. Please input: {}".format(options_to_str(choices))
            )

        # value is always of a string type here
        try:
            value = int(value)
        except ValueError:
            pass

        # value is still a string if it was not converted to int
        if isinstance(value, six.string_types):
            if value not in choices:
                raise ValueError("Invalid input")
        elif isinstance(value, six.integer_types):
            if value < 1 or value > len(choices):
                raise ValueError("Invalid input")
            selected = choices[value - 1]
            return selected
        else:
            raise ValueError("Invalid input")

        return value

    return validator


# can be used to repeat user input
def repeat_input(ns, value):
    print(value)
    return value


# used to unset existing value by typing ""
def not_required_validator(ns, value):
    if value == '""':
        return None
    return value


# used for checking that key parameters don't contain spaces.
# e.g. component name or argument key
def no_spaces_validator(ns, value):
    if " " in value:
        raise ValueError("This parameter can not contain spaces")
    return value


class MainSectionWizard(object):
    """
    PyPsi based wizard to handle input of component's main section parameters.
    """

    def __init__(self, component_info):
        engine_type_header = (
            "Choose engine type:\n" + options_to_str(engine_options) + "\nEngine type"
        )
        language_header = (
            "Choose component language:\n"
            + options_to_str(language_options)
            + "\nLanguage"
        )
        group_header = "Choose group:\n" + options_to_str(group_options) + "\nGroup"
        model_behavior_header = (
            "Choose model behavior:\n"
            + options_to_str(model_behavior_options)
            + "\nModel behavior"
        )
        user_standalone_header = (
            "Choose value:\n" + options_to_str(boolean_options) + "\nUser standalone"
        )
        self.wizard = wiz.PromptWizard(
            name="Component Configuration",
            description="Provide values for component main body attributes",
            steps=[
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_NAME_FIELD,
                    name="Component name",
                    help="Component identifier name, without spaces.",
                    default=component_info.name,
                    validators=(
                        wiz.required_validator,
                        no_spaces_validator,
                        repeat_input,
                    ),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ENGINE_TYPE_FIELD,
                    name=engine_type_header,
                    help="Choose value from the list.",
                    default=component_info.engine_type
                    if component_info.engine_type
                    else EngineType.GENERIC,
                    validators=(choice_validator(engine_options), repeat_input),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_LANGUAGE_FIELD,
                    name=language_header,
                    help="Choose value from the list.",
                    default=component_info.language
                    if component_info.language
                    else ComponentLanguage.PYTHON,
                    validators=(choice_validator(language_options), repeat_input),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_GROUP_FIELD,
                    name=group_header,
                    help="Choose value from the list.",
                    default=component_info.group,
                    validators=(choice_validator(group_options), repeat_input),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_LABEL_FIELD,
                    name="Label",
                    help="Label to be displayed in MCenter UI",
                    default=component_info.label,
                    validators=wiz.required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_DESCRIPTION_FIELD,
                    name="Description",
                    help="Component functionality description. It is displayed in MCenter UI",
                    default=component_info.description,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_VERSION_FIELD,
                    name="Component version",
                    help="Component version number",
                    default=component_info.version
                    if component_info.version
                    else "1.0.0",
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_USER_STAND_ALONE,
                    name=user_standalone_header,
                    help="Component can be standalone or connected",
                    default=component_info.user_standalone
                    if component_info.user_standalone
                    else "False",
                    validators=(choice_validator(boolean_options), repeat_input),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_PROGRAM_FIELD,
                    name="Program file",
                    help="File name of the program to run. (for Python)",
                    default=component_info.program,
                    validators=not_required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_CLASS_FIELD,
                    name="Component class",
                    help="Name of the class that defines component.(For Python: name only; "
                    "For Java: full package name + class name)",
                    default=component_info.component_class,
                    validators=not_required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_MODEL_BEHAVIOR_FIELD,
                    name=model_behavior_header,
                    help="Choose value from the list.",
                    default=component_info.model_behavior
                    if component_info.model_behavior
                    else ComponentModelBehaviorType.MODEL_CONSUMER,
                    validators=choice_validator(model_behavior_options),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_USE_MLOPS_FIELD,
                    name="Use MLOps",
                    help="Whenever MLOps package is used in the component or not: "
                    + options_to_str(boolean_options),
                    default=component_info.use_mlops
                    if component_info.use_mlops
                    else "True",
                    validators=(choice_validator(boolean_options)),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_PYTHON_DEPS,
                    name="Dependencies",
                    help="Dependency packages, comma separated. Example: pytest, numpy, pandas",
                    default=component_info.deps,
                    validators=(not_required_validator, self._deps_validator),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_INCLUDE_GLOB_PATTERNS,
                    name="Include Glob Patterns",
                    help="Example: **/folder | file.txt",
                    default=component_info.include_glob_patterns,
                    validators=not_required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_EXCLUDE_GLOB_PATTERNS,
                    name="Exclude Glob Patterns",
                    help="Example: **/folder | file.txt",
                    default=component_info.exclude_glob_patterns,
                    validators=not_required_validator,
                ),
            ],
        )

    def run(self, shell, print_header):
        return self.wizard.run(shell, print_header=print_header)

    def _deps_validator(self, ns, value):
        if isinstance(value, six.string_types):
            if not len(value):
                return None
            try:
                value = value.replace(" ", "").split(",")
            except:  # noqa: E722
                raise ValueError("Wrong format. Comma separated values expected")
        return value


class ConnectionWizard(object):
    """
    PyPsi based wizard to handle input of component's input/output connections sections.
    """

    def __init__(self, name, comp_conn):
        self.wizard = wiz.PromptWizard(
            name=name,
            description="Provide values for component connection attributes",
            steps=[
                wiz.WizardStep(
                    id=json_fields.CONNECTION_DESC_DESCRIPTION_FIELD,
                    name="Description",
                    help="Connection Description.",
                    default=comp_conn.description,
                    validators=not_required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.CONNECTION_DESC_LABEL_FIELD,
                    name="Label",
                    help="Label to be displayed in MCenter UI",
                    default=comp_conn.label,
                ),
                wiz.WizardStep(
                    id=json_fields.CONNECTION_DESC_TYPE_FIELD,
                    name="Type",
                    help="Type",
                    default=comp_conn.type if comp_conn.type else "str",
                ),
                wiz.WizardStep(
                    id=json_fields.CONNECTION_DESC_GROUP_FIELD,
                    name="Group",
                    help="Group",
                    default=comp_conn.group if comp_conn.group else "data",
                ),
                wiz.WizardStep(
                    id=json_fields.CONNECTION_DESC_DEFAULT_COMPONENT_FIELD,
                    name="Default Component",
                    help="Default Component",
                    default=comp_conn.default_component,
                ),
            ],
        )

    def run(self, shell, print_header):
        return self.wizard.run(shell, print_header=print_header)


class ArgumentWizard(object):
    """
    PyPsi based wizard to handle input of component's arguments section.
    """

    def __init__(self, component_info, arg):
        argument_type_header = (
            "Choose argument type:\n" + options_to_str(argument_type_options) + "\nType"
        )
        argument_tag_header = (
            "Choose argument tag:\n" + options_to_str(argument_tag_options) + "\nTag"
        )
        self.wizard = wiz.PromptWizard(
            name="Argument Configuration",
            description="Provide values for argument attributes",
            steps=[
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_KEY,
                    name="Key",
                    help="Key",
                    default=arg.key,
                    validators=(
                        wiz.required_validator,
                        no_spaces_validator,
                        self._key_existance_validator(
                            component_info, current_key=arg.key
                        ),
                    ),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_TYPE,
                    name=argument_type_header,
                    help="Choose value from the list.",
                    default=arg.type if arg.type else "string",
                    validators=choice_validator(argument_type_options),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_LABEL,
                    name="Label",
                    help="Label",
                    default=arg.label,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_DESCRIPTION,
                    name="Description",
                    help="Description",
                    default=arg.description,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_ENV_VAR,
                    name="Env Var",
                    help="Env Var",
                    default=arg.env_var,
                    validators=not_required_validator,
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_OPTIONAL,
                    name="Optional",
                    help="Is parameter optional",
                    default=arg.optional if arg.optional else "False",
                    validators=(choice_validator(boolean_options)),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_TAG,
                    name=argument_tag_header,
                    help="Tag",
                    default=arg.tag,
                    validators=choice_validator(argument_tag_options),
                ),
                wiz.WizardStep(
                    id=json_fields.COMPONENT_DESC_ARGUMENT_DEFAULT_VAL,
                    name="Default Value",
                    help="Default Value",
                    default=arg.default_value,
                    validators=not_required_validator,
                ),
            ],
        )

    def run(self, shell, print_header):
        return self.wizard.run(shell, print_header=print_header)

    def _key_existance_validator(self, component_info, current_key):
        def validator(ns, value):
            comp_arg = component_info.get_argument(value)

            # When editing argument its key is default value for the step.
            # It should be allowed to keep this key(hit Enter) and not to check if it
            # already exists. In edit mode current_key is not None and new key can be
            # equal to current key.
            if comp_arg is not None and current_key and current_key != comp_arg.key:
                raise ValueError(
                    "Argument with the key '{}' is already defined".format(value)
                )

            return value

        return validator


def path_completer(wizard, token, prefix=""):
    """
    Path completer method for PyPsi PromptWizard
    """
    choices = []
    if not token:
        cwd = "." + os.path.sep
        filename_prefix = ""
    elif token[-1] == os.path.sep:
        cwd = os.path.expanduser(token[:-1])
        filename_prefix = ""
    else:
        token = token[-1]
        filename_prefix = os.path.basename(token)
        cwd = os.path.expanduser(os.path.dirname(token) or "." + os.path.sep)

    if not os.path.exists(cwd):
        return []

    if not os.path.isdir(cwd):
        return []

    for filename in os.listdir(cwd):
        if not filename.startswith(filename_prefix):
            continue

        if os.path.isdir(os.path.join(cwd, filename)):
            filename += os.path.sep
        else:
            filename += "\0"

        filename_len = len(filename_prefix)
        choices.append(prefix + filename[filename_len:])

    return choices


class SaveLoadWizard(object):
    """
    pypsi based wizard to handle input of save/load component json filename
    """

    def __init__(self, name, default_path):

        if name.lower().startswith("save"):
            step = wiz.WizardStep(
                id="path",
                name="File path",
                help="File path to save component file",
                default=default_path,
                validators=(wiz.required_validator, self._save_file_validator),
                completer=path_completer,
            )
            description = "Provide filepath to save component"
        elif name.lower().startswith("load"):
            step = wiz.WizardStep(
                id="path",
                name="File path",
                help="File path to load component",
                default=default_path,
                validators=(wiz.required_validator, self._load_file_validator),
                completer=path_completer,
            )
            description = "Provide component file to load"

        self.wizard = wiz.PromptWizard(name=name, description=description, steps=[step])

    def _save_file_validator(self, ns, value):
        """
        Method to validate value for saving a file
        """
        dir, file = os.path.split(value)

        if dir == "":
            dir = os.getcwd()

        dir = os.path.expanduser(dir)
        filepath = os.path.join(dir, file)

        if not os.path.exists(dir):
            raise ValueError("path '{}' does not exists".format(dir))

        if not os.path.isdir(dir):
            raise ValueError("path '{}' is not directory".format(dir))

        if os.path.exists(filepath):
            raw = ""
            while raw not in ["y", "n"]:
                try:
                    raw = input(
                        "File {} already exists, overwrite? (y/n): ".format(filepath)
                    )
                except (KeyboardInterrupt, EOFError):
                    print()
                    print(AnsiCodes.red, "Wizard canceled", AnsiCodes.reset)
                    return None
            if raw == "n":
                return None

        return filepath

    def _load_file_validator(self, ns, value):
        """
        Method to validate value for loading a file
        """
        dir, file = os.path.split(value)

        if dir == "":
            dir = os.getcwd()

        filepath = os.path.join(dir, file)
        filepath = os.path.expanduser(filepath)

        if not os.path.exists(filepath):
            raise ValueError("path '{}' does not exists".format(filepath))

        if os.path.isdir(filepath):
            raise ValueError("path '{}' is directory".format(filepath))

        return filepath

    def run(self, shell, print_header):
        return self.wizard.run(shell, print_header=print_header)
