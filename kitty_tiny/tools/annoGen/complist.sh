#!/bin/bash


find /raid/templates/farm-data/refine_data/send/01.Deogpung/Key-point/ -newermt 2021-12-15 -name "*.mp4" > sentlist.txt
find /raid/templates/farm-data/refine_data/receive/01.Deogpung/Key-point/ -newermt 2021-12-15 -name "*.mp4" > recvlist.txt

while read line; 
  do 
	  kk=$(basename $line) 
	  # echo $kk; 
	  while read sentline; 
	  do 
	  jj=$(basename $sentline); 

	  if [[ "$kk" == "$jj" ]]; then
		 line=''
	         break;
          fi
		  # echo $jj
		  # if [[ $kk == $jj ]]; then 
	#		  echo "kkkkkkkkkkkkkkkkk" 
	#		  break 
	#	  fi
	  done < sentlist.txt

	  if [[ "$line" != "" ]]; then echo $line; fi

  done < recvlist.txt
