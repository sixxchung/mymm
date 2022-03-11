# python ./my/download_img.py -u ./my/url.txt -o ./my/img_output
import argparse
import os
from os import path as osp

from imutils import paths
import requests
import cv2

parser = argparse.ArgumentParser(description="Download image")
parser.add_argument("-u", "--urls",   ) # required=True, help="URL of img")
parser.add_argument("-o", "--output", ) # required=True, help="output directory of result")
args = parser.parse_args()

# for debug
# args = argparse.Namespace(urls='./url.txt', output='./out')
# args = argparse.Namespace(urls = osp.join(os.getcwd(),'my/url.txt'),
#                           output = osp.join(os.getcwd(), 'my/out'))

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
url_rowList = open(args.urls).read().strip().split("\n")
fileList = os.listdir(args.output)

filenames=[]
for filename in fileList:
    try:
        filenames.append(int(os.path.basename(filename.split('.')[0])))
    except:
        print("")

#total = 0
total = max(filenames)

# loop the URLs
for url in url_rowList:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
		# save the image to disk
		p = os.path.sep.join([args.output, 
            "{}.jpg".format(str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(args.output):
	# initialize if the image should be deleted or not
	delete = False
	# try to load the image
	try:
		image = cv2.imread(imagePath)
		# if the image is `None` then we could not properly load it
		# from disk, so delete it
		if image is None:
			delete = True
	# if OpenCV cannot load the image then the image is likely
	# corrupt so we should delete it
	except:
		print("Except")
		delete = True
	# check to see if the image should be deleted
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)

# /raid/templates/farm-data/car/2nd_dataset
