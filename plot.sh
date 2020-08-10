#!/bin/zsh

for d in ./storage/*/; do
    python plot.py "$d"
done