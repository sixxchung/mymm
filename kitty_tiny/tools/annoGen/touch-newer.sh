#! /bin/bash
echo $1
find /raid/templates/farm-data/refine_data/receive/ -type f -newermt "$1"
