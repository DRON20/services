#!/usr/bin/python

import sys
reload(sys)
import numpy as np
import os
import time
import gc
import random
import caffe
#caffe.set_device(0)
#caffe.set_mode_gpu()
from caffe.proto import caffe_pb2
import ConfigParser
import argparse

caffe_root = '.'

#Read config
def getConfig():
    parser = ConfigParser.ConfigParser()
    config_file = "python/test_classify/classify.ini"
    if (os.path.exists(config_file)):
	parser.read(config_file)
    else:
	print("Config file do not exists")
	sys.exit()
    config = {}
    options = parser.options('default')
    for option in xrange(len(options)):
	config[option] = parser.get('default', options[option])
    return config

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder", type=str, help="path to folder of image")
    parser.add_argument("-mode", type=str, help="choose print format: full or short")
    return parser.parse_args()
    
def getNetwork(model, weights):
    net = caffe.Net(model, weights, caffe.TEST)
    net.blobs['data'].reshape (8, 3, 224, 224)
    return net

def getTransformer(net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer

def getLabels(labels_file):
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    return labels

def print_result(mode, result, time, labels, file_name):
    if mode == 'short':
        print file_name, " ", labels[result.argmax() - 1], " ", result[result.argmax()], " ", time
    if mode == 'full':
        print file_name
        for k in xrange(len(labels)):
            print labels[k], " ", result[k + 1]
        print 'Garbage collector: lvl1 = ', gc.get_count()[0],  " lvl2 = ",  gc.get_count()[1],  " lvl3 = ",  gc.get_count()[2]
        print 'Time =',  time
        print " "

def main(argv):
    args = getArgs()
    config = getConfig()
    net = getNetwork(caffe_root + config[0], caffe_root + config[1])
    transformer = getTransformer(net)
    labels = getLabels(caffe_root + config[2])
    list = os.listdir(args.folder)
    random.shuffle(list)
    for i in list:
        t1 = time.time()
        image = caffe.io.load_image(args.folder + i)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        out = net.forward()
        out_prob = out['prob'][0]
        print_result(args.mode, out_prob, time.time() - t1, labels, i)
    

if __name__ == "__main__":
    main(sys.argv)
