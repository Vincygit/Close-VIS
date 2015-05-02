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

videos = glob.glob("realdata/*.mp4")
common = 0;
# get the real start time
for filename in videos:
    filename, realname = unicodeFilename(filename), filename
    parser = createParser(filename, realname)
    metadata = extractMetadata(parser)
    text = metadata.exportPlaintext()
    for txt in text:
        if "Duration" in txt:
            result = txt.split(" ")
            dura = int(result[2])*60
            if len(result) > 6:
                dura += int(result[4])
                dura += (float(result[6])/1000)
            else:
                dura += (float(result[4])/1000)
        
    print "duration=%f"%dura   
    modlast = os.path.getmtime(filename)
    print "last modified: %f, %s" % (modlast,time.ctime(modlast));
    realStart = modlast - dura; # - (Duration + modlast - common)
    
    if common < realStart:
        common = realStart
        
print "most late starting file time is %f :%s" %(common, time.ctime(common))        
        
# now clip the videos with the data needed
for filename in videos:
    filename, realname = unicodeFilename(filename), filename
    parser = createParser(filename, realname)
    metadata = extractMetadata(parser)
    text = metadata.exportPlaintext()
    for txt in text:
        if "Duration" in txt:
            result = txt.split(" ")
            dura = int(result[2])*60
            if len(result) > 6:
                dura += int(result[4])
                dura += (float(result[6])/1000)
            else:
                dura += (float(result[4])/1000)
        
    print "duration=%f"%dura   
    modlast = os.path.getmtime(filename)
    print "last modified: %f, %s" % (modlast,time.ctime(modlast));
    realStart = modlast - dura; # - (Duration + modlast - common)
    
    nstart = common - realStart
        
    print 'ffmpeg -i %s -ss %d -t %d temp/%s'% (filename, int(nstart), Duration, filename);
    p = subprocess.Popen(['ffmpeg -i %s -ss %d -t %d temp/%s '% (filename, int(nstart), Duration, filename)], shell=True)
    p.wait();
    
    print '%s clipped!' %filename; 

