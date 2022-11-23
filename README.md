# ICF-MI-framework

## Pre-commit
All PR's to this repo must pass formatting and linting checks before they can be merged.
To have these checks performed locally on your commits, run the following commands:

```bash
pip install pre-commit
pre-commit install
```

This will install the pre-commit hooks into your local git repo.  Now, when you commit, the hooks will be run and any errors will be reported.
To manually run the checks, you can run:

```bash
pre-commit run --all-files
```
Running the pre-commit hooks might modify the staged files.
Run the diff command to check the changes made after which the file needs to be staged
again and commited.

## Documentation (For dev)
This repository use Sphinx for documenting the code.
To install sphinx and the necessary requirements run,

```bash
./scripts/configure_sphinx.sh
```

To document your function use the following docsting,

```bash

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

```

Now, to update the documentation run,

```
./scripts/create_documentation.sh
```
from the **ROOT** of the project.
If everything works perfectly, a a firefox window will pop-up with the documentation.
Please verify that the desired documentation has been made.
