#!/usr/bin/env bash

echo "========== Downloading all files =========="
wget -i files.txt -nc

# extract test and dev files, they end up in the right dir hierarchy
echo "========== Extracting test and dev set =========="
for file in dev.tgz test-filtered.tgz
do
	tar -xvf $file
done

mkdir train

# common crawl
echo "========== Extracting commoncrawl dataset =========="
mkdir train/commoncrawl
tar -C train/commoncrawl -xvf training-parallel-commoncrawl.tgz

# europarl, news commentary and UN corpus
for name in europarl-v7 nc-v9 un
do
	echo "========== Extracting ${name} dataset =========="
	mkdir train/$name
	tar -C train/$name -xvf training-parallel-$name.tgz
	# unnest dataset
	mv train/$name/**/* train/$name
done

# giga fr-en
echo "========== Extracting gigafren dataset =========="
mkdir train/gigafren
tar -C train/gigafren -xvf training-giga-fren.tar
cd train/gigafren
gunzip -v *.gz
cd ../..

find . -type d -empty -delete
