import argparse 
import os        # output폴더만들기...

parser = argparse.ArgumentParser(description='Argument Parse Test Code')
# argument는 원하는 만큼 추가한다.
parser.add_argument('--print_number', type=int,    help='an integer for printing repeatably')
parser.add_argument("-u", "--url", required=False, help="youtube url")
args = parser.parse_args()
# args = parser.parse_known_args()
# args = argparse.Namespace(print_number=6)  # for debug
# args = argparse.Namespace(url='https://youtu.be/NBoPIS9bAuU')  # for debug

for i in range(args.print_number):
    print(f'print number {format(i+1)}')