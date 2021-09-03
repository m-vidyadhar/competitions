#!/bin/bash

echo 'Running: ' pythonw train.py -e 40 -l 1e-3 -t 0 -d 1
pythonw train.py -e 40 -l 1e-3 -t 0 -d 1 -p 10 -o 1 -k 3

echo 'Testing the best model: ' 
pythonw train.py -t 1