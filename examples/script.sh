#!/bin/bash
python3 examples/script.py factor pd
seq 0 14 | xargs -n 1 -P 15 python3 examples/script.py partial pd
python3 examples/script.py final pd