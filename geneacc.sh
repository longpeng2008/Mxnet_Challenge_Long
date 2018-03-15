grep "top_k_accuracy_3" log.txt | grep "Speed" | grep "Batch \[320"  > acc_train.log
grep "Validation-top_k_accuracy_3" log.txt > acc_val.log


