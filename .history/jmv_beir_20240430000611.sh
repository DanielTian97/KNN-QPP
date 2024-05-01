#!/bin/bash

if [ $# -lt 5 ]
then
        echo "Usage: $0 <msmarco res> <beir res> <ap/ndcg> <nqc/uef> <rbo?> <extend?>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.TRECDLQPPEvaluatorBEIR" -Dexec.args="$1 $2 $3 $4 $5 $6"
