#!/bin/bash
for i in {1..10}
do
   echo "creating $i";
#    out_file="1M_$i";
   log_file="logs/transfer_test_${i}.txt";
   dd if=/dev/zero of= $log_file bs=1024 count=1;
done
# dd if=/dev/zero of=testfile bs=1024 count=1024000
# dd if=/dev/zero of=testfile bs=1024 count=1024000
# dd if=/dev/zero of=testfile bs=1024 count=1024000