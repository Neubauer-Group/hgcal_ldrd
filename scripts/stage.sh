#!/usr/bin/env bash
mkdir -p $2/raw
rm $2/raw/*
cp $1/*_$3.npz $2/raw

