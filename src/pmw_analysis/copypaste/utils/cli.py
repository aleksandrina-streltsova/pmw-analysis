import argparse
import enum

# copied from https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    # If you prefer to specify the enum by name change:
    #
    #     tuple(e.value for e in enum_type) to tuple(e.name for e in enum_type)
    #     value = self._enum(values) to value = self._enum[values]
    #
    # Using the name also allows for the use of IntEnum and auto().
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)

