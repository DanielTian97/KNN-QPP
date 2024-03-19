FILE=$1
a=`egrep "Train on.*.2019" $FILE | head -n1 | awk '{print $NF}'`
b=`egrep "Train on.*.2020" $FILE | head -n1 | awk '{print $NF}'`
echo "$a $b" | awk '{print "NQC: " ($1+$2)/2}'
grep -P "Target.*.tau = " $FILE
grep "Target" $FILE | awk '{print "JM: " $NF}'