import argparse 
import os

import pytube # pip install git+https://github.com/baxterisme/pytube

parser = argparse.ArgumentParser(description="Download youtube for data gathering")
parser.add_argument("-u", "--url", help="URL of youtube")  #  required=True,
args = parser.parse_args()

# for debug
#args = argparse.Namespace(url='https://youtu.be/NBoPIS9bAuU')

streamList = pytube.YouTube(args.url).streams

result_stream = []
keywords=["720p", "1080p", "1440p"]
for stream in streamList:
    if 'video/mp4' in str(stream):
        if any(keyword  in str(stream) for keyword in keywords):
            result_stream.append(stream)

for idx, stream in enumerate(result_stream):
    print(f" {idx} :: {stream}")

s = input('press number key to select video index :  ')
result_stream[int(s)].download()

