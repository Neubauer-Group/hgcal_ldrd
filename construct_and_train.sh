#!/bin/#!/usr/bin/env bash

echo Activating Conda Environment
conda activate hgcal-env

gnn_path="/data/gnn_code/"
training_path=$gnn_path"training_data/"
heptrkx_path=$gnn_path"heptrkx-gnn-tracking/"
hgcal_path=$gnn_path"hgcal_ldrd/"
config_file="configs/"$1".yaml"
echo Constructing graph using $heptrkx_path$config_file

cd $heptrkx_path
python prepare.py $config_file --n-workers=24

echo Moving Graph Files
cd $training_path
rm -rf $1
mkdir $1
mkdir $1/processed
mkdir $1/test
mkdir $1/raw
cd $1/raw
mv $heptrkx_path/output/*_g000.npz .
mv $heptrkx_path/output/*_g001.npz .

cd $training_path
rm -rf test_convert
mkdir test_convert
mkdir test_convert/processed
mkdir test_convert/raw
cd test_convert/raw
mv $training_path$1/raw/event0000099* .
rm *_g001.npz

echo converting test graphs
cd $hgcal_path
source env.sh
python scripts/convert_test_graphs.py -c -m=EdgeNetWithCategories -l=nll_loss -d=test_convert --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW

cd $training_path$1/test
mv $training_path/test_convert/processed/* .
cd $training_path
rm -rf test_convert

echo Training GNN
cd $hgcal_path
source env.sh
python scripts/heptrx_nnconv.py -c -m=EdgeNetWithCategories -l=nll_loss -d=$1 --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW |& tee EdgeNetWithCategories_$1.log

cd $gnn_path
