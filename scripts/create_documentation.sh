#!/bin/bash
# This script creates the documentation for the project.
# Run this script from the root of the project.

cd docs
echo $PWD

rm `find . -name "*rst" ! -name "index.rst" ! -name "introduction.rst"`
sphinx-apidoc -o . ../
make html
make latexpdf
firefox ./_build/html/index.html
