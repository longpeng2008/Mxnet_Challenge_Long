i=0
#array=(224 256 288 320 352 384 416 488 480 512)
array=(224 256 288 320 352 384 416 488 480 512)
for data in ${array[@]}
do
   shape=${data}
   for k in $( seq 30 30 )
   do
      echo $k
      python evaluate.py $k 0 model/dpn92-365std 30 test_results/submit_""$k""_10_singlescale_meanmaxcrop_""$shape"".json $shape test_results/probs/prob_""$k""_10_singlescale_meanmaxcrop_""$shape"".txt
   done
done
