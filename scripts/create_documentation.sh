#!/bin/bash
# This script creates the documentation for the project.
# Run this script from the root of the project.

cd docs
echo $PWD

sphinx-apidoc -o . ../
make html
make latexpdf
firefox ./_build/html/index.html
