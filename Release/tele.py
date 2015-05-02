#!/usr/bin/python
import os.path, time
import glob
import time as time_ #make sure we don't override time
import subprocess
from subprocess import PIPE
from hachoir_core.error import HachoirError
from hachoir_core.cmd_line import unicodeFilename
from hachoir_parser import createParser
from hachoir_core.tools import makePrintable
from hachoir_metadata import extractMetadata
from hachoir_core.i18n import getTerminalCharset
from sys import argv, stderr, exit

Duration = 300;
videos=glob.glob("/home/vincy/research/videos/*.mp4")
common = time_.time();
sa=""
for i in videos:
    end = os.path.getmtime(i)
    col = os.path.getctime(i)
    print "create:%s, end:%s"%(col,end)
    print "time last %s :%f "%(i, end-col)
    if common > end:
        common = end
        sa = i


# now clip the videos with the data needed
#for fname in videos:
fname=videos[1]
start = os.path.getctime(fname)
nstart = common-start- Duration;
print "start@@@@@%f" %(start)
print "nstart@@@@@%f" %(nstart)
if nstart < 0:
    print "%s"%(fname)
#print 'ffmpeg -i %s -ss %s -t %s 5m%s'% (fname, nstart, Duration, fname);
#p = subprocess.Popen(['ffmpeg -i %s -ss %s -t %s 5m%s'% (fname, nstart, Duration, fname)], shell=True)
#print "%d has been executed" % p.pid
#
        
#     if(commonEndding > end)
#         common = end
#     print "created: %s" % time.ctime(os.path.getctime(i))
