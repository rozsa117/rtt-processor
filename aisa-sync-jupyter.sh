#!/bin/bash
SDIR="$(SHELL_SESSION_FILE='' && cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rsync -av --progress aisa:~/RTT/proc*.ipynb ${SDIR}
