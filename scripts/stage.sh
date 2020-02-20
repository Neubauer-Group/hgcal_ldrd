#!/usr/bin/env bash
mkdir -p $2/test_track/raw
rm $2/test_track/raw/*
cp $1/*_$3.npz $2/test_track/raw

