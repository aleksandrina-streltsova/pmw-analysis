#!/bin/bash
python3 examples/script.py factor pd
seq 0 5 | xargs -n 1 -P 6 python3 examples/script.py partial pd
python3 examples/script.py final pd