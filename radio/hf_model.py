from transformers.dynamic_module_utils import get_class_from_dynamic_module
import os
import sys


class RADIOConfig:  # type: ignore
    def __new__(cls, *args, **kwargs):
        # Prefer local Radiov2 package if present
        radiov2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Radiov2"))
        if os.path.isdir(radiov2_dir) and radiov2_dir not in sys.path:
            sys.path.insert(0, radiov2_dir)
        try:
            from Radiov2.hf_model import RADIOConfig as LocalRADIOConfig  # type: ignore
            remote_cls = LocalRADIOConfig
        except Exception as e:
            raise ImportError(f"Local Radiov2 not found or failed to import: {e}")
        instance = object.__new__(remote_cls)
        remote_cls.__init__(instance, *args, **kwargs)
        return instance


class RADIOModel:  # type: ignore
    def __new__(cls, *args, **kwargs):
        # Prefer local Radiov2 package if present
        radiov2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Radiov2"))
        if os.path.isdir(radiov2_dir) and radiov2_dir not in sys.path:
            sys.path.insert(0, radiov2_dir)
        try:
            from Radiov2.hf_model import RADIOModel as LocalRADIOModel  # type: ignore
            remote_cls = LocalRADIOModel
        except Exception as e:
            raise ImportError(f"Local Radiov2 not found or failed to import: {e}")
        instance = object.__new__(remote_cls)
        remote_cls.__init__(instance, *args, **kwargs)
        return instance


