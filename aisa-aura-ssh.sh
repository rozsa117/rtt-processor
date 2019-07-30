#!/bin/bash
: "${JPORT:=8870}"
ssh -L $JPORT:localhost:$JPORT -A aisa -t "ssh -L $JPORT:localhost:$JPORT aura -t 'jupyter notebook list; bash -l'"
