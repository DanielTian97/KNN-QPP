#!/bin/bash

if [ $# -lt 6 ]
then
        echo "Usage: $0 <dl19 res> <dl20 res> <ap/ndcg> <nqc/uef> <sbert/rlm/w2v> <extend_1?> <useRBO?>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.TRECDLQPPEvaluatorWithGenVariants" -Dexec.args="$1 $2 $3 $4 $5 $6 $7"
