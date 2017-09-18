#!/usr/bin/python
# -*- coding: utf-8 -*-
#from __future__ import unicode_literals

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import time
import threading
from PIL import Image as img
import os
caffe_root = '.'
import gc
import logging
import cgi
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import ConfigParser
import tempfile as tmp

# Чтение из  ini файла
def getConfig ():
	parser = ConfigParser.ConfigParser()
	if (os.path.exists("ssd_server.ini")):
		parser.read("ssd_server.ini")
	else:
		print("ssd_server.ini don`t exists")
		sys.exit()
	config = {}
	options = parser.options('default')
	for option in xrange(len(options)):
		config[option] = parser.get('default', options[option])
	return config

config = getConfig()
model = None
weight = None
labelmap_file = None
i = 0
net = 0
tmp_net = 0
N = 1000

dataFileNo = 0
module = 256

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', filename='ssd_server.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

def createNet():
        global tmp_net
        tmp_net = initNet(namespace.model, namespace.weight)

class HttpProcessor(BaseHTTPRequestHandler):

    def do_POST(self):
        t1 = time.time()
        global log
        global N
        global i
        global tmp_net
        global net
        i += 1
        logging.debug('Number of forward = %s', i)
        if(i > N)and(tmp_net != 0):
                t2 = time.time()
                net = tmp_net
                i = 0
                tmp_net = 0
                logging.debug('Exchange = %s', str(time.time() - t2))
        if(i == N):
                time1 = time.time()
                t = threading.Thread(target=createNet, args=())
                t.daemon = True
                t.start()
                logging.debug('ReInit = %s', time.time() - time1)
        time1 = time.time()
        # 1
        length = int(self.headers['Content-Length'])
        form = cgi.FieldStorage(
            fp = self.rfile,
            headers = self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        time2 = time.time()
        # 2
        data = form['fileToUpload'].file.read()
        
        global dataFileNo
        dataFileNo = (dataFileNo + 1) % module
        dataFileName = '/tmp/ssd/data-' + str(dataFileNo)
        imgFile = open(dataFileName,'w')
        imgFile.write(data)
        imgFile.close()
        
        time3 = time.time()
        # 3
        prediction = str(getPredictions(data))
        logging.debug('Returns: %s', prediction)
        time4 = time.time()
        # 4
        sendResult(self, prediction)
        logging.debug('do_POST = 1: %s, 2: %s, 3: %s, 4: %s', time2 - time1, time3 - time2, time4 -  time3, time.time() -  time4)
        logging.debug('All do_post = %s', str(time.time() - t1))
        logging.debug('Garbage collector: lvl1 = %s, lvl2 = %s, lvl3 = %s', gc.get_count()[0], gc.get_count()[1], gc.get_count()[2])

    def do_GET(self):
        print('This method do not exist')


def sendResult(self, resultStr):
    self.send_response(200)
    self.send_header("Content-Type", 'text/plain; charset=iso-8859-1')    
    count = len(resultStr)
    self.send_header("Content-Length", str(count))
    self.end_headers()
    self.wfile.write(resultStr)
    self.wfile.write('\n')


def get_labelname(labelmap, labels):
          num_labels = len(labelmap.item)
          labelnames = []
          if type(labels) is not list:
              labels = [labels]
          for label in labels:
              found = False
              for i in xrange(0, num_labels):
                  if label == labelmap.item[i].label:
                      found = True
                      labelnames.append(labelmap.item[i].display_name)
                      break
              assert found == True
          return labelnames

def getPredictions(imagefile):
    global net
    # Prepare
    file = open(labelmap_file)
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    file.close()

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    t = tmp.NamedTemporaryFile(delete=False, suffix=".bmp", prefix="image")
    t.write(imagefile)
    t.close()
    image_resize = 300
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    image = caffe.io.load_image(t.name, True)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    pilimage = img.open(t.name, 'r')
    WxH = pilimage.size
    
    # Net.forward
    detections = net.forward()['detection_out']
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    
    # Data
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    out = []
    for i in xrange(len(top_labels)): 
	if (((top_xmax[i] - top_xmin[i]) * WxH[0] < WxH[0]) or ((top_ymax[i] - top_ymin[i]) * WxH[1] < WxH[1])):
        	out.append(' '.join([top_labels[i], str(top_conf[i]), str(int(top_xmin[i] * WxH[0])), str(int(top_ymin[i] * WxH[1])), str(int(top_xmax[i] * WxH[0])), str(int(top_ymax[i] * WxH[1]))]))
    return '\n'.join(out)
    


def initNet(model, weight):
    network = caffe.Net(model, weight, caffe.TEST)
    logging.info('========= Net init =========')
    return network

def init(port, model, weight):
    global net
    net = initNet(model, weight)

    logging.info('port = %s', port)
    logging.info('model = %s', model)
    logging.info('weight = %s', weight)
    logging.info('label_map = %s', labelmap_file)

    logging.info('=========Starting server=========')
    serv = HTTPServer(("0.0.0.0", port), HttpProcessor)
    serv.serve_forever()

def main(argv):
    global labelmap_file
    model = caffe_root + config[0]
    weight = caffe_root + config[1]
    labelmap_file = caffe_root + config[2]
    port = int(config[3])

    init(port, model, weight)


if __name__ == "__main__":
    main(sys.argv)
