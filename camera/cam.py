from pitop import Camera,Pitop
import cv2

camera = Camera()
pitop = Pitop()
frame_count=0

while True:
	frame = camera.get_frame()
	pitop.miniscreen.display_image(frame)

	filename= f"frame{frame_count}.jpg"
	cv2.imwrite (filename,frame)

	frame_count +=1
