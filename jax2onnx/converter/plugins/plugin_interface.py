# plugin_interface.py
from typing import Protocol, Callable, Any, TypeVar

# Let T be the type of the converter; you can later import the actual type.
T = TypeVar("T", contravariant=True)


class Plugin(Protocol[T]):
    @staticmethod
    def get_primitive() -> Any:
        """
        Return the JAX primitive that this plugin handles.
        """

    @staticmethod
    def get_handler(s: T) -> Callable[[Any, Any, dict], None]:
        """
        Return a handler function for the primitive.
        The handler should accept (node_inputs, node_outputs, params).
        """

    @staticmethod
    def get_metadata() -> dict:
        """
        Return the metadata dictionaries for this plugin.
        """
