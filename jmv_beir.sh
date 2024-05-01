#!/bin/bash

if [ $# -lt 6 ]
then
        echo "Usage: $0 <msmarco res> <beir res> <ap/ndcg> <nqc/uef> <sbert/rlm/w2v> <extend_1?> <useRBO?>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.TRECDLQPPEvaluatorGenBEIR" -Dexec.args="$1 $2 $3 $4 $5 $6 $7"
