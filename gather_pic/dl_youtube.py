import argparse
import pytube # pip install pytube
import os

parser = argparse.ArgumentParser(description="Download youtube for data gathering by sixx")
parser.add_argument("-u", "--url", help="URL of youtube")  #  required=True,
args = parser.parse_args()
# args = parser.parse_known_args()
# args = argparse.Namespace(url='https://youtu.be/NBoPIS9bAuU')  # for debug

streamList = pytube.YouTube(args.url).streams

result_stream = []
keywords=["720p", "1080p", "1440p"]
for stream in streamList:
    if not 'video/mp4' in str(stream):
        continue
    elif any(keyword  in str(stream) for keyword in keywords):
        result_stream.append(stream)
for idx, stream in enumerate(result_stream):
    print(f" {idx} :: {stream}")

s = input('press number key to select video index :  ')
result_stream[int(s)].download()

# $ python dl_youtube.py --url https://youtu.be/NBoPIS9bAuU