#!/usr/bin/env bash

python3 -m pip install --upgrade pip
pip install flake8 pytest

if [ -f requirements.txt ]; then 
    pip install -r requirements.txt; 
fi

mkdir temp
cp -R src/* temp/
cp -R tests/* temp/
cd temp
pytest . 
cd ..
rm -r temp