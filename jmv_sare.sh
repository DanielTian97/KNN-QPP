#!/bin/bash

if [ $# -lt 8 ]
then
        echo "Usage: $0 <dl19 res> <dl20 res> <ap/ndcg> <nqc/uef> <sbert/bm25/sbertr/bm25r--(qv_method)> <extend_1?> <k> <lambda>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.TRECDLQPPEvaluatorWithGenVariantsSARE" -Dexec.args="$1 $2 $3 $4 $5 $6 $7 $8"
