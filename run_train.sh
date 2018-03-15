export MXNET_CPU_WORKER_NTHREADS=160
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py --pretrained-model model/dpn92-365std \
    --load-epoch 0 --gpus 0,1,2,3 \
    --data-train data/list_ai_train_all_shuffled_prepared.txt --model-prefix model/dpn92-365std \
    --data-val data/list_ai_validation_all_prepared.txt \
	--data-nthreads 128 \
    --batch-size 160 --num-classes 80 --num-examples 53879 2>&1 | tee log.txt

