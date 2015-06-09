#!/usr/bin/env bash

filenames=$(ls /mnt/data/sunlight/openstates_zipped_files/)

for i in $filenames; do
	dir_name=$(sed -E 's/201[0-9]-0[0-9]-0[0-9]-//g' ${i} | sed -E 's/-json.zip//g')
	unzip /mnt/data/sunlight/openstates_zipped_files/${i} -d /mnt/data/sunlight/openstates_unzipped/${dir_name}
done
