from pypsi import wizard as wiz
from mlpiper.cli.wizard_shell import CommandName, EXIT_CODE


# TODO:
# make connection type an options list

# remove header, customize?
# steps dependency
# ask for dir and generate __init__ and main.py with class

# validate file is component json
# if quit - exit totally
# Discuss: are these required: label, description


class States:
    START = "start"
    FINISH = "finish"
    ADD_COMPONENT = "add_component"
    ADD_INPUT = "add_input"
    ADD_ADDITIONAL_INPUT = "add_additional_input"
    ADD_OUTPUT = "add_output"
    ADD_ADDITIONAL_OUTPUT = "add_additional_output"
    ADD_ARGUMENT = "add_argument"
    ADD_ADDITIONAL_ARGUMENT = "add_additional_argument"
    SAVE = "save"


class State(object):
    class YesNoWizard(object):
        def __init__(self, name="yesno", header=""):
            self.wizard = wiz.PromptWizard(
                name=name,
                description="Provide values for component connection attributes",
                steps=[
                    wiz.WizardStep(
                        id="answer",
                        name=header,
                        help="Answer Yes or No",
                        default="Yes",
                        validators=wiz.boolean_validator,
                    )
                ],
            )

        def run(self, shell):
            return self.wizard.run(shell, print_header=False)

    def __init__(self, header=""):
        self._header = header

    def _yes_no_wizard(self, shell, header):
        ns = State.YesNoWizard(name="yesno", header=header).run(shell)
        return ns.answer if ns else False

    def run(self, shell, step_input):
        raise Exception("Not implemented")


class StartState(State):
    def run(self, shell):
        return States.ADD_COMPONENT


class FinishState(State):
    pass


class AddComponentState(State):
    def run(self, shell):
        ret_code = shell.execute(CommandName.COMPONENT)
        if ret_code == EXIT_CODE:
            return States.FINISH
        return States.ADD_INPUT


class AddInput(State):
    def run(self, shell):
        step_input = self._yes_no_wizard(shell, self._header)
        if step_input:
            shell.execute(CommandName.INPUT)
            return States.ADD_ADDITIONAL_INPUT
        else:
            return States.ADD_OUTPUT


class AddOutput(State):
    def run(self, shell):
        step_input = self._yes_no_wizard(shell, self._header)
        if step_input:
            shell.execute(CommandName.OUTPUT)
            return States.ADD_ADDITIONAL_OUTPUT
        else:
            return States.ADD_ARGUMENT


class AddArgument(State):
    def run(self, shell):
        step_input = self._yes_no_wizard(shell, self._header)
        if step_input:
            shell.execute(CommandName.ARGUMENT)
            return States.ADD_ADDITIONAL_ARGUMENT
        else:
            return States.SAVE


class SaveState(State):
    def run(self, shell):
        shell.execute(CommandName.SAVE)
        return States.FINISH


class WizardFlowStateMachine(object):
    def __init__(self, shell=None):
        self._shell = shell

        self._states = {
            States.START: StartState(),
            States.FINISH: FinishState(),
            States.ADD_COMPONENT: AddComponentState(),
            States.ADD_INPUT: AddInput("Do you want to add an input connection?"),
            States.ADD_ADDITIONAL_INPUT: AddInput(
                "Do you want to add an additional input connection?"
            ),
            States.ADD_OUTPUT: AddOutput("Do you want to add an output connection?"),
            States.ADD_ADDITIONAL_OUTPUT: AddOutput(
                "Do you want to add an additional output connection?"
            ),
            States.ADD_ARGUMENT: AddArgument("Do you want to add an argument?"),
            States.ADD_ADDITIONAL_ARGUMENT: AddArgument(
                "Do you want to add an additional argument?"
            ),
            States.SAVE: SaveState(CommandName.SAVE),
        }

        self._currentState = self._states[States.START]

    def run(self):
        while not isinstance(self._currentState, FinishState):
            out_state = self._currentState.run(self._shell)
            self._currentState = self._states[out_state]
