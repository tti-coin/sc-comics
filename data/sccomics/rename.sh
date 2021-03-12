#!/bin/bash

for f in `find . -name "*.ann"`
    do
        bname=`basename $f`

	    after=`expr ${bname%.*} + 200`
        mv $bname `printf %04d $after`.ann

        #mv $bname "0"$bname
    done

