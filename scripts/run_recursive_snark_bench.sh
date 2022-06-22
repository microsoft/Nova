#!/bin/bash

cd ../

cargo bench --bench recursive-snark > scripts/recursive-snark.txt

cd scripts

python3 raw_to_csv_recursive_snark.py
