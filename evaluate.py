#coding=utf8
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx
import random
import importlib
def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def oversample(images, crop_dims):

    im_shape = np.array(images.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # print crops_ix

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    # for im in images:
    im = images
    # print im.shape
    for crop in crops_ix:
        # print crop
        crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
        # cv2.imshow('crop', im[crop[0]:crop[2], crop[1]:crop[3], :])
        # cv2.waitKey()
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]
    # cv2.imshow('crop', crops[0,:,:,:])
    # cv2.waitKey()
    return crops

#prefix = 'model/Scence-resnet-152-365'
prefix = sys.argv[3]
epoch = int(sys.argv[1]) #check point step
gpu_id = int(sys.argv[2]) #GPU ID for infer
test_iters = int(sys.argv[4]) #test iter times
ctx = mx.gpu(gpu_id)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
#mx.viz.plot_network(sym).view()

###---修改网络结构---###
all_layers = sym.get_internals()
#print "------all layers:",all_layers
sym = all_layers['conv5_x_x__relu-sp__relu_output']
#print "------sym before pool5:",sym
sym  = mx.symbol.Pooling(data=sym, pool_type='avg', kernel=(7,7), stride=(1,1), pad=(0,0), name='avg_pool')
sym = mx.symbol.Convolution(data=sym, num_filter=80, kernel=(1,1),no_bias=False,name='fc')
arg_params['fc_weight'] = arg_params['fc_weight'].reshape(arg_params['fc_weight'].shape+(1,1))
arg_params['fc_bias'] = arg_params['fc_bias'].reshape(arg_params['fc_bias'].shape)

crop_sz = int(sys.argv[6])
mean_max_pool_shape=int(crop_sz)/32-6
mean_max_pooling_size = tuple([mean_max_pool_shape,mean_max_pool_shape])
print "------mean max pooling size:",crop_sz," to ",mean_max_pooling_size
sym1 = mx.symbol.Flatten(data=mx.symbol.Pooling(data=sym, pool_type='avg',kernel=mean_max_pooling_size,stride=(1,1),pad=(0,0),name='out_pool1'))
sym2 = mx.symbol.Flatten(data=mx.symbol.Pooling(data=sym, pool_type='max',kernel=mean_max_pooling_size,stride=(1,1),pad=(0,0),name='out_pool2'))

sym  = (sym1 + sym2 ) / 2.0
sym  = mx.symbol.SoftmaxOutput(data = sym, name = 'softmax_new')
#mx.viz.plot_network(sym).view()


IMAGE_DIR="data/scene_validation_20170908/"
ann_file = 'data/scene_validation_20170908.json'
print('Loading annotations from: ' + os.path.basename(ann_file))
with open(ann_file) as data_file:
    ann_data = json.load(data_file)

imgs = [aa['image_id'] for aa in ann_data]

classes = [0]*len(imgs)

top1_acc = 0
top5_acc = 0
cnt = 0
img_szs = [crop_sz + 32]

fprob = open(sys.argv[7], 'w')
result = []

for index in range(0, len(imgs)):
        img_name = imgs[index]
        label = str(classes[index])
        cnt += 1
        img_full_name = IMAGE_DIR + img_name
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = np.float32(img)
        rows, cols = img.shape[:2]
        avg_score = np.zeros( (80,) ) 

	for img_sz in img_szs:
	   if cols < rows:
              resize_width = img_sz
	      resize_height = resize_width * rows / cols;
           else:
              resize_height = img_sz
              resize_width = resize_height * cols / rows;

           img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
	   img_flipped = cv2.flip(img,1)
           h, w, _ = img.shape

           ####------------use ten random crops----###

	   x0s = [0, 0, w-crop_sz, w-crop_sz, (w-crop_sz)/2]
           y0s = [0, h-crop_sz, 0, h-crop_sz, (h-crop_sz)/2]

	   ##-------origin 
           for i in range(0,5):
              x0 = x0s[i]
              y0 = y0s[i]
              img_crop = img[y0:y0+crop_sz, x0:x0+crop_sz]

              img_crop = np.swapaxes(img_crop, 0, 2)
              img_crop = np.swapaxes(img_crop, 1, 2)  # change to r,g,b order

              img_crop = img_crop[np.newaxis, :]

              arg_params["data"] = mx.nd.array(img_crop, ctx)
              #arg_params["data"] = mx.nd.array(input_blob, ctx)
              arg_params["softmax_new_label"] = mx.nd.empty((1,), ctx)
              exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
              exe.forward(is_train=False)
              probs = exe.outputs[0].asnumpy()
              score = np.squeeze(probs.mean(axis=0))
              avg_score = avg_score + score
              #print "avg_score=",avg_score
	
	   ##-------hozizon flipped 
	   for i in range(0,5):
              x0 = x0s[i]
              y0 = y0s[i]
              img_crop = img_flipped[y0:y0+crop_sz, x0:x0+crop_sz]

              img_crop = np.swapaxes(img_crop, 0, 2)
              img_crop = np.swapaxes(img_crop, 1, 2)  # change to r,g,b order

              img_crop = img_crop[np.newaxis, :]
	   
              arg_params["data"] = mx.nd.array(img_crop, ctx)
              arg_params["softmax_new_label"] = mx.nd.empty((1,), ctx)

              exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
              exe.forward(is_train=False)
              probs = exe.outputs[0].asnumpy()
              score = np.squeeze(probs.mean(axis=0))
              avg_score = avg_score + score
 
        sort_index = np.argsort(avg_score)[::-1]
        top_k = sort_index[0:3]
	#print(top_k)
        temp_dict = {}

        temp_dict['label_id'] = top_k.tolist()
        temp_dict['image_id'] = img_name
        result.append(temp_dict)

	###-----store all the probs-----###
	thisprob = {}
	thisprob['image_id'] = img_name
	out_score=[0]*80
        for i in range(0,len(avg_score)):
	   out_score[i] = avg_score[i] / (10*len(img_szs))
	thisprob['probs'] = out_score
        json.dump(thisprob, fprob)
	fprob.write('\n')


with open(sys.argv[5], 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result))

fprob.close()



