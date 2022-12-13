def templateFunction(arg1, arg2):
    """Summary line.

    Extended description of function.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        arg1 (float): Description of arg1
        arg2 (float): Description of arg2

    Returns:
        (float): Sum of arg1 and arg2

    """

    m = arg1 + arg2

    return m


class main:
    def __init__(
        self,
        data,
        epsilon: float = None,
        keepData: bool = False,
        samplesAlongLastDimension: bool = True,
        method: str = "ttsvd",
    ):
        """This is the main function (LSTM)"""
        print(templateFunction(1, 2))
