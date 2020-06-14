# USAGE
# python3 detect_blinks.py --video blinkvideo.mp4

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd
from deepface import DeepFace
import os
import speech_recognition as sr
from textblob import TextBlob
# from moviepy.editor import *


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def eye_counter(filename):
	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold
	EYE_AR_THRESH = 0.35
	EYE_AR_CONSEC_FRAMES = 3

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0

	final_df = pd.DataFrame(columns=['video_time_stamp', 'blink_counter', 'Emotion','list_of_emotions'])

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	# vs = FileVideoStream(args["video"]).start()
	fileStream = True
	vs = cv2.VideoCapture(filename)
	# fileStream = False
	time.sleep(1.0)
	total_frames_video = 0

	fps = vs.get(cv2.CAP_PROP_FPS)
	print(fps)

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		# if fileStream and not vs.more():
		# 	break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		ret, frame = vs.read()
		if ret ==  False:
			break
		width,height = frame.shape[0], frame.shape[1]
		size = (width,height)
		# frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)




		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1


			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1

				# reset the eye frame counter
				COUNTER = 0

			if total_frames_video % round(fps) == 0:
				print('true')
				emotion = DeepFace.analyze(frame, ['emotion'])
				video_time = (total_frames_video/round(fps))
				video_time = time.strftime('%H:%M:%S', time.gmtime(video_time))
				print(video_time)
				final_df = final_df.append({"video_time_stamp":video_time, "blink_counter":TOTAL, "Emotion":emotion["dominant_emotion"], "list_of_emotions":emotion}, ignore_index = True)

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# # show the frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF
		#
		# # if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break

		total_frames_video+=1


		final_df.to_csv('eye_blinker.csv')

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.release()

def video_to_audio_conversion(audiofile):
	AUDIO_FILE = (audiofile)
	print(AUDIO_FILE)

	r = sr.Recognizer()

	with sr.AudioFile(AUDIO_FILE) as source:
	    #reads the audio file. Here we use record instead of
	    #listen
	    audio = r.record(source)
	try:
	    with open("converted_text.txt", 'w+') as f:
	        f.write(r.recognize_google(audio))
	    # print("The audio file contains: " + r.recognize_google(audio))

	except sr.UnknownValueError:
	    print("Google Speech Recognition could not understand audio")

	except sr.RequestError as e:
	    print("Could not request results from Google Speech Recognition service; {0}".format(e))


	with open("converted_text.txt", 'r') as ff:
	    lines = ff.read()
	    print(lines)

	blob = TextBlob(lines)
	print(blob.sentiment)
	print(blob.sentiment.polarity)

if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--shape-predictor", required=True,
	# 	help="path to facial landmark predictor")
	ap.add_argument("-v", "--video", type=str, default="",
		help="path to input video file")
	args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	video_file = args["video"]

	name_file = video_file.split(".")[0]

	# audio_clip = AudioFileClip(args["video"])

	#
	# command2mp3 = "ffmpeg -i {} {}.mp3".format(video_file,name_file)
	command2wav = "ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav".format(video_file,name_file)


	# os.system(command2mp3)
	os.system(command2wav)

	eye_counter(video_file)

	# video_to_audio_conversion(audio_clip)
	video_to_audio_conversion("{}.wav".format(name_file))
