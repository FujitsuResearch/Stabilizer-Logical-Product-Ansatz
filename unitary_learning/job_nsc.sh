#!/bin/bash

for s in $(seq 1 20)
do
nohup python main_finite_nsc.py -d 8 -ntrain 50 -ntest 50 -nepoch 500 -nsamp 1000 -b 1 -ns 1 -e 0.001 -s $s &
done
