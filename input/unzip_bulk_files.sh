#!/usr/bin/env bash

filenames=$(ls /mnt/data/sunlight/openstates_zipped_files/)

for i in $filenames; do
	unzip $i -d /mnt/data/sunlight/openstates_unzipped/
done
