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


def main():
    """This is the main function (TT-ICE)"""
    print(templateFunction(1, 2))


if __name__ == "__main__":
    main()
