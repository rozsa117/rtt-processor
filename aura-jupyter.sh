#!/bin/bash
# jupyter notebook password
# jupyter notebook list
: "${JPORT:=8870}"
jupyter-notebook  --no-browser --port $JPORT .
