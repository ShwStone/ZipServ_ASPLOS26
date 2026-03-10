#!/bin/bash

M=28672
K=4096
N=32

function test() {
    ./test_mm $M $K $N $1 > ./result_${M}_${K}_${N}_${1}.log
}

test 1
test 2
test 4
test 8
test 16
test 32
test 64
