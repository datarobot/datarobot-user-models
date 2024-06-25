from pypsi.shell import Shell
from pypsi.core import Command, PypsiArgParser, CommandShortCircuit
from pypsi.commands.help import HelpCommand
from pypsi.commands.exit import ExitCommand
from pypsi.ansi import AnsiCodes

from mlpiper.pipeline.component_info import ComponentInfo, ComponentInfoEncoder
from mlpiper.pipeline.component_connection_info import ComponentConnectionInfo
from mlpiper.pipeline.component_argument_info import ComponentArgumentInfo
from mlpiper.cli.wizard_modules import (
    MainSectionWizard,
    ConnectionWizard,
    ArgumentWizard,
    SaveLoadWizard,
)

import six
import json
import io

EXIT_CODE = 127


class CommandName:
    COMPONENT = "comp"
    ARGUMENT = "argument"
    INPUT = "input"
    OUTPUT = "output"
    SHOW = "show"
    SAVE = "save"
    LOAD = "load"
    HELP = "help"


class Helper(object):
    @staticmethod
    def filter_non_zero_string_items(in_dict):
        out_dict = {}
        for k, v in in_dict.items():
            if isinstance(v, six.string_types) and len(v) == 0:
                continue
            out_dict[k] = v
        return out_dict


class CompCommand(Command):
    def __init__(self, name, topic, brief, **kwargs):
        self.parser = PypsiArgParser(prog=name, description=brief)

        super(CompCommand, self).__init__(
            name=name,
            usage=self.parser.format_help(),
            topic=topic,
            brief=brief,
            **kwargs
        )

    def run(self, shell, args):
        try:
            self.parser.parse_args(args)
        except CommandShortCircuit as e:
            return e.code

        component_info = shell._component_info
        # read print_header from args
        ns = MainSectionWizard(component_info).run(shell, print_header=True)
        if ns:
            out_dict = Helper.filter_non_zero_string_items(ns.__dict__)
            component_info.load_from_json(out_dict)
        else:
            return EXIT_CODE

        return 0


# TODO use args parser for sub commands
class InputOutputCommand(Command):
    SUBCOMMAND_DELETE = "delete"
    SUBCOMMAND_EDIT = "edit"
    SUBCOMMAND_ADD = "add"

    def __init__(self, name, topic, brief, **kwargs):
        self.parser = PypsiArgParser(prog=name, description=brief)
        # TODO: start using args from ArgParser, now they only added to show pretty help
        self.parser = PypsiArgParser(prog=name, description=brief)
        subparsers = self.parser.add_subparsers(dest="subparser_name", help="Commands")
        subparsers.add_parser("add", help="add connection")
        edit_arg_parser = subparsers.add_parser("edit", help="edit connection")
        edit_arg_parser.add_argument(
            "index", metavar="INDEX", action="store", help="connection index to edit"
        )
        delete_arg_parser = subparsers.add_parser("delete", help="delete connection")
        delete_arg_parser.add_argument(
            "index", metavar="INDEX", action="store", help="connection index to delete"
        )

        super(InputOutputCommand, self).__init__(
            name=name,
            usage=self.parser.format_help(),
            topic=topic,
            brief=brief,
            **kwargs
        )

        self._handlers = {
            self.SUBCOMMAND_ADD: self._add_handler,
            self.SUBCOMMAND_DELETE: self._delete_handler,
            self.SUBCOMMAND_EDIT: self._edit_handler,
        }
        self._print_header = True

    def run(self, shell, args):
        try:
            self.parser.parse_args(args)
        except CommandShortCircuit as e:
            return e.code

        component_info = shell._component_info
        self._print_header = shell._wizard_edit_mode
        if self.name == CommandName.INPUT:
            if component_info._inputs is None:
                component_info._inputs = []
            self._inputs_or_outputs_array = component_info._inputs
        elif self.name == CommandName.OUTPUT:
            if component_info._outputs is None:
                component_info._outputs = []
            self._inputs_or_outputs_array = component_info._outputs

        cmd = self.SUBCOMMAND_ADD
        if args is not None and len(args):
            if args[0] not in self._handlers.keys():
                self.usage_error(shell, args)
                return 1

            cmd = args[0]

        return self._handlers[cmd](shell, args)

    def _add_handler(self, shell, args=None):
        comp_conn = ComponentConnectionInfo()
        wizard_header = "{} Connection Configuration".format(self.name.capitalize())
        ns = ConnectionWizard(wizard_header, comp_conn).run(shell, self._print_header)
        if ns:
            out_dict = Helper.filter_non_zero_string_items(ns.__dict__)
            comp_conn.load_from_json(out_dict)
            self._inputs_or_outputs_array.append(comp_conn)
        return 0

    def _edit_handler(self, shell, args):
        if len(args) < 2:
            self.usage_error(shell, args)
            return 1
        try:
            index = int(args[1])
        except:  # noqa: E722
            print("Index parameter '{}' can not be casted to int".format(args[1]))
            self.usage_error(shell, args)
            return 0

        if 0 <= index < len(self._inputs_or_outputs_array):
            comp_conn = self._inputs_or_outputs_array[index]
            wizard_header = "{} Connection Configuration".format(self.name.capitalize())
            ns = ConnectionWizard(wizard_header, comp_conn).run(
                shell, self._print_header
            )
            if ns:
                out_dict = Helper.filter_non_zero_string_items(ns.__dict__)
                comp_conn.load_from_json(out_dict)
        else:
            print("Element with index '{}' is not found".format(index))
            self.usage_error(shell, args)

        return 0

    def _delete_handler(self, shell, args):
        if len(args) < 2:
            self.usage_error(shell, args)
            return 1
        try:
            index = int(args[1])
        except:  # noqa: E722
            print("Index parameter '{}' can not be casted to int".format(args[1]))
            self.usage_error(shell, args)
            return 0

        if 0 <= index < len(self._inputs_or_outputs_array):
            self._inputs_or_outputs_array.pop(index)
        else:
            print("Element with index '{}' is not found".format(index))
            self.usage_error(shell, args)
        return 0


class ArgumentCommand(Command):
    SUBCOMMAND_DELETE = "delete"
    SUBCOMMAND_EDIT = "edit"
    SUBCOMMAND_ADD = "add"

    def __init__(self, name, topic, brief, **kwargs):

        # TODO: start using args from ArgParser, now they only added to show pretty help
        self.parser = PypsiArgParser(prog=name, description=brief)
        subparsers = self.parser.add_subparsers(dest="subparser_name", help="Commands")
        subparsers.add_parser("add", help="add argument")
        edit_arg_parser = subparsers.add_parser("edit", help="edit argument")
        edit_arg_parser.add_argument(
            "key", metavar="KEY", action="store", help="argument key to edit"
        )
        delete_arg_parser = subparsers.add_parser("delete", help="delete argument")
        delete_arg_parser.add_argument(
            "key", metavar="KEY", action="store", help="argument key to delete"
        )

        super(ArgumentCommand, self).__init__(
            name=name,
            usage=self.parser.format_help(),
            topic=topic,
            brief=brief,
            **kwargs
        )

        self._handlers = {
            self.SUBCOMMAND_ADD: self._add_handler,
            self.SUBCOMMAND_DELETE: self._delete_handler,
            self.SUBCOMMAND_EDIT: self._edit_handler,
        }

    def run(self, shell, args):
        try:
            self.parser.parse_args(args)
        except CommandShortCircuit as e:
            return e.code

        cmd = self.SUBCOMMAND_ADD
        self._print_header = shell._wizard_edit_mode
        if args is not None and len(args):
            if args[0] not in self._handlers.keys():
                self.usage_error(shell, args)
                return 1

            cmd = args[0]

        return self._handlers[cmd](shell, args)

    def _add_handler(self, shell, args=None):
        component_info = shell._component_info

        comp_arg = ComponentArgumentInfo()
        ns = ArgumentWizard(component_info, comp_arg).run(shell, self._print_header)
        if ns:
            out_dict = Helper.filter_non_zero_string_items(ns.__dict__)
            comp_arg.load_from_json(out_dict)

            if component_info.arguments is None:
                component_info.arguments = []
            component_info.arguments.append(comp_arg)
        return 0

    def _edit_handler(self, shell, args):
        if len(args) < 2:
            self.usage_error(shell, args)
            return 1
        component_info = shell._component_info
        key = args[1]
        comp_arg = component_info.get_argument(key)
        if comp_arg:
            ns = ArgumentWizard(component_info, comp_arg).run(shell, self._print_header)
            if ns:
                out_dict = Helper.filter_non_zero_string_items(ns.__dict__)
                comp_arg.load_from_json(out_dict)
        else:
            print("Argument with the key '{}' is not found".format(key))
            self.usage_error(shell, args)

        return 0

    def _delete_handler(self, shell, args):
        if len(args) < 2:
            self.usage_error(shell, args)
            return 1
        component_info = shell._component_info
        key = args[1]
        arg = component_info.get_argument(key)
        if arg:
            component_info.arguments.remove(arg)
        else:
            print("Argument with the key '{}' is not found".format(key))
            self.usage_error(shell, args)
        return 0


class ShowCommand(Command):
    def __init__(self, name, topic, brief, **kwargs):

        self.parser = PypsiArgParser(prog=name, description=brief)
        super(ShowCommand, self).__init__(
            name=name,
            usage=self.parser.format_help(),
            topic=topic,
            brief=brief,
            **kwargs
        )

    def run(self, shell, args):
        try:
            self.parser.parse_args(args)
        except CommandShortCircuit as e:
            return e.code

        component_info = shell._component_info
        print(json.dumps(component_info, indent=4, cls=ComponentInfoEncoder))

        return 0


class SaveLoadCommand(Command):
    def __init__(self, name, topic, brief, **kwargs):
        self.parser = PypsiArgParser(prog=name, description=brief)
        super(SaveLoadCommand, self).__init__(
            name=name,
            usage=self.parser.format_help(),
            topic=topic,
            brief=brief,
            **kwargs
        )

    def run(self, shell, args):
        try:
            self.parser.parse_args(args)
        except CommandShortCircuit as e:
            return e.code

        if self.name == CommandName.SAVE:
            w = SaveLoadWizard("Save Component", shell._filename)
            ns = w.run(shell, True)
            if ns:
                if ns.path is not None:
                    component_info = shell._component_info
                    with io.open(ns.path, mode="w", encoding="utf-8") as f:
                        json.dump(component_info, f, indent=4, cls=ComponentInfoEncoder)
                    print("Component was successfully saved: {}".format(ns.path))
                    shell._filename = ns.path
                else:
                    print("Abort saving component")
        elif self.name == CommandName.LOAD:
            w = SaveLoadWizard("Load Component", None)
            ns = w.run(shell, True)
            if ns:
                with io.open(ns.path, encoding="utf-8") as f:
                    try:
                        json_dict = json.load(f)
                        shell._component_info.load_from_json(json_dict)
                    except Exception as e:
                        print(
                            "Can not load file {}. Not a json file?. Error: {}".format(
                                ns.path, str(e)
                            )
                        )
                        return 1
                shell._filename = ns.path
                print("Component was successfully loaded: {}".format(ns.path))
        else:
            pass

        return 0


class ComponentWizardShell(Shell):

    # add commands to the shell
    topic = "Component Builder Commands"
    print_cmd = ShowCommand(
        name=CommandName.SHOW, brief="show current component json", topic=topic
    )
    save_cmd = SaveLoadCommand(
        name=CommandName.SAVE, brief="save current component json to file", topic=topic
    )
    load_cmd = SaveLoadCommand(
        name=CommandName.LOAD, brief="load component from json file", topic=topic
    )

    comp_wiz = CompCommand(
        name=CommandName.COMPONENT,
        brief="add or edit component's main body",
        topic=topic,
    )
    input_conn_wiz = InputOutputCommand(
        name=CommandName.INPUT,
        brief="add, edit or delete component input connection",
        topic=topic,
    )
    output_conn_wiz = InputOutputCommand(
        name=CommandName.OUTPUT,
        brief="add, edit or delete component output connection",
        topic=topic,
    )
    argument_wiz = ArgumentCommand(
        name=CommandName.ARGUMENT,
        brief="add, edit or delete component argument",
        topic=topic,
    )

    service_topic = "Service commands"
    help_cmd = HelpCommand(topic=service_topic)
    exit_cmd = ExitCommand(topic=service_topic)

    def __init__(self, shell_name, wizard_edit_mode):
        super(ComponentWizardShell, self).__init__(shell_name=shell_name)

        self.prompt = "{cyan}component builder{r} {green}>{r} ".format(
            r=AnsiCodes.reset.prompt(),
            cyan=AnsiCodes.cyan.prompt(),
            green=AnsiCodes.green.prompt(),
        )

        self._component_info = ComponentInfo()
        self._wizard_edit_mode = wizard_edit_mode
        self._filename = None

    def on_cmdloop_begin(self):
        print(AnsiCodes.clear_screen)
        self.execute(CommandName.HELP)
