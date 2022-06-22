#!/bin/bash

cd ../

cargo bench --bench compressed-snark > scripts/compressed-snark.txt

cd scripts

python3 raw_to_csv_compressed_snark.py
