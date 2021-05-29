# import the necessary packages
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import os


class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

# load our serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join(['hed_model',
	"deploy.prototxt"])
modelPath = os.path.sep.join(['hed_model',
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

def auto_canny(image, sigma=0.7):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

class CardExtractor:
	def __init__(self) -> None:
		pass

	def extract(self, img):
		# load the query image, compute the ratio of the old height
		# to the new height, clone it, and resize it
		image = cv2.imread(img)
		ratio = image.shape[0] / 600.0
		orig = image.copy()
		image = imutils.resize(image, height = 600)
		(H, W) = image.shape[:2]

		# construct a blob out of the input image for the Holistically-Nested
		# Edge Detector
		blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
			swapRB=False, crop=False)

		# set the blob as the input to the network and perform a forward pass
		# to compute the edges
		print("[INFO] performing holistically-nested edge detection...")
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (W, H))
		hed = (255 * hed).astype("uint8")
		
		# hed = cv2.adaptiveThreshold(hed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) 
		cv2.imwrite("edged.jpg", hed)

		canny = cv2.Canny(hed, 5, 30, L2gradient=False)
		cv2.imwrite("canny.jpg", canny)


		# find contours in the edged image, keep only the largest
		# ones, and initialize our screen contour
		cnts = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		# allCnts = image.copy()
		# cv2.drawContours(allCnts, cnts, -1, (0, 255, 0), thickness=1)
		# cv2.imwrite('all.jpg', allCnts)
		# screenCnt = None
		# rectangles = []
		# loop over our contours
		# for c in cnts:
		# 	# approximate the contour
		# 	peri = cv2.arcLength(c, True)
		# 	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		# 	rectangles.append(approx)
		# 	# if our approximated contour has four points, then
		# 	# we can assume that we have found our screen
		# 	if len(approx) == 4:
		# 		screenCnt = approx
		# 		break


		allCnts = image.copy()
		cv2.drawContours(allCnts, cnts, -1, (0, 255, 0), thickness=2)
		cv2.imwrite('all.jpg', allCnts)

		# biggestArea = image.copy()
		# cv2.drawContours(biggestArea, cnts, 0, (0, 255, 0), thickness=2)
		# cv2.imwrite('biggest.jpg', biggestArea)

		# # draw a rectangle around the screen
		peri = cv2.arcLength(cnts[0], True)
		screenCnt = cv2.approxPolyDP(cnts[0], 0.04 * peri, True)
		cv2.drawContours(image, cnts, 0, (0, 255, 0), thickness=2)
		cv2.imwrite("found.jpg", image)

		# now that we have our screen contour, we need to determine
		# the top-left, top-right, bottom-right, and bottom-left
		# points so that we can later warp the image -- we'll start
		# by reshaping our contour to be our finals and initializing
		# our output rectangle in top-left, top-right, bottom-right,
		# and bottom-left order
		pts = screenCnt.reshape(4, 2)
		rect = np.zeros((4, 2), dtype = "float32")

		# the top-left point has the smallest sum whereas the
		# bottom-right has the largest sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]

		# compute the difference between the points -- the top-right
		# will have the minumum difference and the bottom-left will
		# have the maximum difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]

		# multiply the rectangle by the original ratio
		rect *= ratio

		# now that we have our rectangle of points, let's compute
		# the width of our new image
		(tl, tr, br, bl) = rect
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

		# ...and now for the height of our new image
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

		# take the maximum of the width and height values to reach
		# our final dimensions
		maxWidth = max(int(widthA), int(widthB))
		maxHeight = max(int(heightA), int(heightB))

		# construct our destination points which will be used to
		# map the screen to a top-down, "birds eye" view
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		# calculate the perspective transform matrix and warp
		# the perspective to grab the screen
		M = cv2.getPerspectiveTransform(rect, dst)
		warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

		# convert the warped image to grayscale and then adjust
		# the intensity of the pixels to have minimum and maximum
		# values of 0 and 255, respectively
		# warpAdaptive = run_histogram_equalization(warp)
		# warp1 = exposure.rescale_intensity(warp, out_range = (0, 255))

		cv2.imwrite(img, warp)
		return warp