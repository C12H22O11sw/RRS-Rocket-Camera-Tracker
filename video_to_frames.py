import cv2
import numpy as np



import cv2
vidcap = cv2.VideoCapture('data/trevor_rocket_1.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("data/trevor_rocket_1/frame%d.jpg" % count, image)     # save frame as JPEG file
  print(count)
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1
