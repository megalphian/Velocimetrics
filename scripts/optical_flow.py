
#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from time import clock
from math import *
from data_processing import *

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

_Plotting = plotting()

class App:
    def __init__(self, video_src, graph_on):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        global cascade_src
        cascade_src = '../data/cars.xml'
        self.good_speed = None
        self._speed = None
        self.car_each_frame = []
        self.car_counter = 0
        self.graph_on = graph_on

        if self.graph_on:
            _Plotting.launch_plt()


    def run(self):
    	global counter
    	global temp
    	car_cascade = cv2.CascadeClassifier(cascade_src)
    	temp = []
    	counter = 0
    	fps = self.cam.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) :",fps)
        while True:
            cam_ret, cam_frame = self.cam.read()
    	    if cam_frame is None:
    	        break
    	    size_param = cam_frame.shape
    	    frame = cam_frame[size_param[0]/2:size_param[0], 0:size_param[1]];
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
    	    cv2.GaussianBlur(vis, (15, 15), 0)
    	    cars = car_cascade.detectMultiScale(vis, 1.1, 1)
            cv2.putText(vis, "cars detected", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness = 2)
            cv2.putText(vis, str(len(cars)), (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness = 3)

            _Plotting.xdata.append(self.frame_idx)
            _Plotting.ydata.append(len(cars))

            if self.graph_on:
                _Plotting.on_running()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
		        #print ("Data", tr)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                self._speed = (p1-p0)*fps
                self.good_speed = self._speed < 0

                #print(len(self._speed), len(self.tracks))

                #print ("Speed = ",self._speed)
            	new_tracks = []

                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                for i in range(0,len(self.good_speed)):
        			if self.good_speed[i][0] is True:
        				temp = [self._speed[i][0]]
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

    	    for (x,y,w,h) in cars:
            	cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
                count = 0
                speed_sum = 0
                pt_in = 0
                t = clock()
                for pts in self.tracks:
                    if x < pts[0][0] < (x + w) and y < pts[0][1] < (y + h):
                        pt_in = pt_in + 1
                        cv2.circle(vis, (pts[0][0], pts[0][1]), 3, (0, 0, 255), 2)
                        speed_temp = sqrt( pow(self._speed[count][0][0],2) + pow(self._speed[count][0][1],2))
                        speed_sum = speed_sum + speed_temp
                    count  = count + 1
                if pt_in > 0 and clock() - t < 0.025:
                    cv2.putText(vis, str(round(speed_sum/pt_in, 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness = 2)

            if self.frame_idx % self.detect_interval == 0:
                #counter +=1
                #print ("Counter ", counter)
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            #print ("Frame_IDX", self.frame_idx)
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cam_frame[size_param[0]/2:size_param[0], 0:size_param[1]] = vis;
            cv2.imshow('lk_track', cam_frame)
            ch = cv2.waitKey(30)
            #raw_input("Press Enter to continue...")
            if ch == 27:
                    break


def main():
    import sys
    try:
        video_src = sys.argv[1]
        graph_switch = sys.argv[2]
        _Plotting.graph_on = int(graph_switch)
    except:
        video_src = 0

    print(__doc__)
    App(video_src, int(graph_switch)).run()
    #digits_video.main()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
