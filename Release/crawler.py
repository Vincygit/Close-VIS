
import subprocess
from subprocess import PIPE
import time

# the address of the server, do not edit this value
videoHost = "rtmp://wfs.495express.mp.advection.net/event/495express/liveroad/"

# for more cameras, add the camera number below
cameraId=(
'AID-495-N-45.1-1',
'AID-495-N-47.4-1',
'AID-495-N-47.4-2',
'AID-495-N-49.7-1',
'AID-495-N-49.7-2',
'AID-495-N-51.5-1',
'AID-495-N-51.5-2',
'AID-495-N-52.3-1',
'AID-495-N-52.3-2',
'AID-495-N-53.4-1',
'AID-495-N-53.4-2',
'AID-495-S-45.9-1',
'AID-495-S-46.2-1',
'AID-495-S-46.7-1',
'AID-495-S-46.7-2',
'AID-495-S-48.8-1',
'AID-495-S-49.0-1',
'AID-495-S-50.5-1',
'AID-495-N-50.5-2',
'AID-495-S-52.9-1',
'AID-495-S-52.9-2',
'AID-495-S-54.8-1',
'AID-495-S-54.8-2',
  
  
'AID-95-S-167.1-2', 
'AID-95-N-155.2',
'AID-95-S-156.9',
'AID-95-S-157.2',
'AID-95-N-158.6-1', 
'AID-95-N-145.5',
'AID-95-S-151.8',
'AID-95-S-161.1',
'AID-95-S-162.1',
'AID-95-N-162.3',
'AID-95-S-162.6',
'AID-95-S-165.9',
'AID-95-N-166.3',
'AID-95-N-167.9',
'AID-95-S-168.1',
'AID-95-S-168.8-2', 
'AID-95-N-169.3',
'AID-95-N-169.6',
'AID-95-S-169.8',
'AID-95-N-170.0',
'AID-95-S-170.6')


processes = []
for i in range(len(cameraId)):
    p = subprocess.Popen(['ffmpeg -i %s%s.stream -c copy %s.mp4'% (videoHost, cameraId[i], cameraId[i])], shell=True)
    processes.append(p)
    print "%d:%s has been executed" % (p.pid,cameraId[i])

# change this value for recording time.
# note that you should set a higher value than the time you wanted
# because the execution of rtmpdump may consume extra time.
time.sleep(1000)

subprocess.Popen.terminate()


