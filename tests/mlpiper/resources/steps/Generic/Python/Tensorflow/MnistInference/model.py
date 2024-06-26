import abc
from future.utils import with_metaclass
import subprocess


class Model(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, model_dir, signature_def):
        self._model_dir = model_dir
        self._signature_def = signature_def
        self._input_name = ""
        self._output_name = ""
        self._output_shape = ""
        self.parse_sig()

    @abc.abstractmethod
    def infer(self, sample, label):
        """Should return sample, label, inference."""
        pass

    def get_signature_name(self):
        return self._signature_def

    def get_input_name(self):
        return self._input_name

    def get_output_name(self):
        return self._output_name

    def get_output_shape(self):
        return self._output_shape

    def parse_sig(self):
        result = subprocess.check_output(
            ["saved_model_cli", "show", "--dir", self._model_dir, "--all"]
        )
        desired_sig = "signature_def['{}']:".format(self._signature_def)
        sig = "signature_def"
        lines = result.split("\n")
        i = 0
        found = False

        for line in lines:
            if line == desired_sig:
                found = True
                break
            if line.startswith(sig):
                print("Found signature: {}".format(line.replace("signature_def", "")))
            i += 1

        if found:
            # Assumes only 1 input tensor and 1 output tensor
            for i in range(i + 1, len(lines) - 1):
                line = lines[i].strip()
                if line.startswith(sig):
                    break
                if line.startswith("inputs"):
                    self._input_name = lines[i + 3].strip().replace("name: ", "")
                elif line.startswith("outputs"):
                    # Assume output is flattened
                    self._output_shape = int(
                        lines[i + 2]
                        .strip()
                        .replace("shape: (-1, ", "")
                        .replace(")", "")
                    )
                    self._output_name = lines[i + 3].strip().replace("name: ", "")

        else:
            raise Exception("Desired signature not found: ", self._signature_def)

    def __del__(self):
        pass
