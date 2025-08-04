#!/bin/bash
data_folder=$1
find $data_folder -type f ! -name 'BOUT.inp' -delete