#!/bin/bash

if [ $# -lt 2 ]
then
        echo "Usage: $0 <retriever> <queryset> <qppmethod>"
        exit
fi

mvn exec:java -Dexec.mainClass="experiments.QPPEvaluatorSimple" -Dexec.args="$1 $2 $3"
