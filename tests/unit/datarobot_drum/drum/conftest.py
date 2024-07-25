from typing import Optional

from datarobot_drum.drum.enum import CustomHooks

from datarobot_drum.drum.adapters.model_adapters.python_model_adapter import PythonModelAdapter


class ChatPythonModelAdapter(PythonModelAdapter):
    chat_hook = None

    def __init__(self, model_dir, target_type):
        super().__init__(model_dir, target_type)

        self._custom_hooks[CustomHooks.CHAT] = self._call_chat_hook

    def load_model_from_artifact(
        self,
        user_secrets_mount_path: Optional[str] = None,
        user_secrets_prefix: Optional[str] = None,
        skip_predictor_lookup=False,
    ):
        return ""

    def _call_chat_hook(self, model, completion_create_params):
        return ChatPythonModelAdapter.chat_hook(model, completion_create_params)
