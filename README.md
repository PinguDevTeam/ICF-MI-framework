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
