#!/bin/bash

if [ $# -lt 9 ]
then
        echo "Usage: $0 <dl19 res> <dl20 res> <ap/ndcg> <nqc/uef> <sbert/bm25/sbertr/bm25r--(qv_method)> <extend_1?> <p> <k> <lambda>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.TRECDLQPPEvaluatorWithGenVariantsKShotLlamaSARE" -Dexec.args="$1 $2 $3 $4 $5 $6 $7 $8 $9"
