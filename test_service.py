#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests
import os
import sys
from datetime import datetime as dt
import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-url", type=str, help="server`s url address without http:// ")
    parser.add_argument("-file", type=str, help="path to file")
    return parser.parse_args()
args = getArgs()
try:
    headers = {'Content-Type':'image/jpg', 'Content-Length':str(os.path.getsize(args.file))}
    r = requests.post(str("http://" + args.url),
                  files={'fileToUpload':open(args.file, 'r')});
    print r.content
except Exception as error:
    file = open('log.txt', 'a')
    file.write(str(dt.now()) + ' ')
    file.write(args.file + ' ')
    file.write(str(error) + '\n')
    file.close()
    print str(error)
