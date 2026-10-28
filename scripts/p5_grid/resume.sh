#!/bin/bash
# resume.sh [card ...]   (no cards = everything)
R=/opt/dlami/nvme/kvcache
if [ $# -eq 0 ]; then rm -f $R/PAUSE_ALL $R/PAUSE_CARD_*; else for c in "$@"; do rm -f $R/PAUSE_CARD_$c; done; fi
echo "resumed: ${*:-all}; still paused: $(ls $R/PAUSE_* 2>/dev/null | sed 's|.*/||' | tr '\n' ' ')"
