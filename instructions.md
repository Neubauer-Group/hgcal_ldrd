# Instructions for running the GNN code on the BEAST

Here are links to the original repositories and files:

Lindsey Gray's hgcal code. I have been using the branch named *gravnet2_wip_trainer_args*
[https://github.com/lgray/hgcal_ldrd](https://github.com/lgray/hgcal_ldrd)
This is located in **/data/gnn_code/hgcal_ldrd**  

And the heptrkx code.
[https://github.com/esaliya/heptrkx-gnn-tracking](https://github.com/esaliya/heptrkx-gnn-tracking)
This is located in **/data/gnn_code/heptrkx-gnn-tracking**

The TrackML data can be downloaded from the following [https://www.kaggle.com/c/trackml-particle-identification/data](https://www.kaggle.com/c/trackml-particle-identification/data)
There is a copy of the data located at **/data/gnn_code/trackml-particle-identification**

There is also a sample root file that I got from Lindsey. This is CALO data from CMS. It is located at **/data/gnn_code/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_1.root**

<br>

## Python Environment

The first thing that needs to be done, is you will need to create the correct python environment in order to execute the code. Lindsey has some instructions on how to do this, but following his instructions took me a while. I dumped a copy of my environment to the following location: **/data/gnn_code/hgcal_ldrd/hgcal-env.yml**

In order to create your own copy of the environment you need to run the following command
```
  conda env create -f /data/gnn_code/hgcal_ldrd/hgcal-env.yml
```

To open the conda environment type the following
```
conda activate hgcal-env
```

Here is a list of the packages that are in the environment that are needed
```
conda list

# packages in environment at /home/markus/miniconda3/envs/hgcal-env:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
attrs                     19.3.0                     py_0  
backcall                  0.1.0                    py36_0  
blas                      1.0                         mkl  
bleach                    3.1.0                    py36_0  
ca-certificates           2020.1.1                      0  
certifi                   2019.11.28               py36_0  
cffi                      1.13.2           py36h2e261b9_0  
chardet                   3.0.4                    pypi_0    pypi
cudatoolkit               10.0.130                      0  
cycler                    0.10.0                   py36_0  
dbus                      1.13.12              h746ee38_0  
decorator                 4.4.1                      py_0  
defusedxml                0.6.0                      py_0  
entrypoints               0.3                      py36_0  
expat                     2.2.6                he6710b0_0  
fontconfig                2.13.0               h9420a91_0  
freetype                  2.9.1                h8a8886c_1  
glib                      2.63.1               h5a9c865_0  
gmp                       6.1.2                h6c8ec71_1  
googledrivedownloader     0.4                      pypi_0    pypi
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb453b48_1  
h5py                      2.10.0                   pypi_0    pypi
icu                       58.2                 h9c2bf20_1  
idna                      2.8                      pypi_0    pypi
importlib_metadata        0.23                     py36_0  
intel-openmp              2019.4                      243  
ipykernel                 5.1.3            py36h39e3cac_0  
ipython                   7.9.0            py36h39e3cac_0  
ipython_genutils          0.2.0                    py36_0  
ipywidgets                7.5.1                      py_0  
isodate                   0.6.0                    pypi_0    pypi
jedi                      0.15.1                   py36_0  
jinja2                    2.10.3                     py_0  
jpeg                      9b                   h024ee3a_2  
jsonschema                3.2.0                    py36_0  
jupyter                   1.0.0                    py36_7  
jupyter_client            5.3.4                    py36_0  
jupyter_console           6.1.0                      py_0  
jupyter_core              4.6.1                    py36_0  
kiwisolver                1.1.0            py36he6710b0_0  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libpng                    1.6.37               hbc83047_0  
libsodium                 1.0.16               h1bed415_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h2733197_0  
libuuid                   1.0.3                h1bed415_2  
libxcb                    1.13                 h1bed415_1  
libxml2                   2.9.9                hea5a465_1  
markupsafe                1.1.1            py36h7b6447c_0  
matplotlib                3.1.1            py36h5429711_0  
mistune                   0.8.4            py36h7b6447c_0  
mkl                       2019.4                      243  
mkl-service               2.3.0            py36he904b0f_0  
mkl_fft                   1.0.15           py36ha843d7b_0  
mkl_random                1.1.0            py36hd6b4f25_0  
more-itertools            7.2.0                    py36_0  
nbconvert                 5.4.1                    py36_3  
nbformat                  4.4.0                    py36_0  
ncurses                   6.1                  he6710b0_1  
networkx                  2.4                      pypi_0    pypi
ninja                     1.9.0            py36hfd86e86_0  
notebook                  6.0.2                    py36_0  
numpy                     1.17.4           py36hc1035e2_0  
numpy-base                1.17.4           py36hde5b4d6_0  
olefile                   0.46                     py36_0  
openssl                   1.1.1d               h7b6447c_3  
pandas                    0.25.3           py36he6710b0_0  
pandoc                    2.2.3.2                       0  
pandocfilters             1.4.2                    py36_1  
parso                     0.5.1                      py_0  
pcre                      8.43                 he6710b0_0  
pexpect                   4.7.0                    py36_0  
pickleshare               0.7.5                    py36_0  
pillow                    6.2.1            py36h34e0f95_0  
pip                       19.3.1                   py36_0  
plyfile                   0.7.1                    pypi_0    pypi
prometheus_client         0.7.1                      py_0  
prompt_toolkit            2.0.10                     py_0  
ptyprocess                0.6.0                    py36_0  
pybasic                   0.31.dev0                pypi_0    pypi
pycparser                 2.19                     py36_0  
pygments                  2.4.2                      py_0  
pyparsing                 2.4.5                      py_0  
pyqt                      5.9.2            py36h05f1152_2  
pyrsistent                0.15.6           py36h7b6447c_0  
python                    3.6.9                h265db76_0  
python-dateutil           2.8.1                      py_0  
pytorch                   1.2.0           py3.6_cuda10.0.130_cudnn7.6.2_0    pytorch
pytz                      2019.3                     py_0  
pyyaml                    5.3              py36h7b6447c_0  
pyzmq                     18.1.0           py36he6710b0_0  
qt                        5.9.7                h5867ecd_1  
qtconsole                 4.6.0                      py_0  
rdflib                    4.2.2                    pypi_0    pypi
readline                  7.0                  h7b6447c_5  
requests                  2.22.0                   pypi_0    pypi
send2trash                1.5.0                    py36_0  
setuptools                42.0.1                   py36_0  
sip                       4.19.8           py36hf484d3e_0  
six                       1.13.0                   py36_0  
sqlite                    3.30.1               h7b6447c_0  
terminado                 0.8.3                    py36_0  
testpath                  0.4.4                      py_0  
tk                        8.6.8                hbc83047_0  
torch-cluster             1.4.5                    pypi_0    pypi
torch-geometric           1.3.2                    pypi_0    pypi
torch-scatter             1.4.0                    pypi_0    pypi
torch-sparse              0.4.3                    pypi_0    pypi
torch-spline-conv         1.1.1                    pypi_0    pypi
torchvision               0.4.0                py36_cu100    pytorch
tornado                   6.0.3            py36h7b6447c_0  
tqdm                      4.40.0                     py_0    conda-forge
trackml                   0.1.12                   pypi_0    pypi
traitlets                 4.3.3                    py36_0  
unionfind                 0.0.10                   pypi_0    pypi
urllib3                   1.25.7                   pypi_0    pypi
wcwidth                   0.1.7                    py36_0  
webencodings              0.5.1                    py36_1  
websocket-client          0.57.0                   pypi_0    pypi
wheel                     0.33.6                   py36_0  
widgetsnbextension        3.5.1                    py36_0  
xgboost                   0.90                     pypi_0    pypi
xz                        5.2.4                h14c3975_4  
yaml                      0.1.7                had09818_2  
zeromq                    4.3.1                he6710b0_3  
zipp                      0.6.0                      py_0  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0  

```

If for some reason the environment doesnt work with the provided .yml file, this is the list of packages that you will need to get into your environment.

<br>

## Preprocessing the TrackML data

This step is straight forward, but can be time consuming. This step uses the heptrkx code. There is a config file that you will need to edit before running the preprocessing. An example of a config file that I have been using is located at **/data/gnn_code/heptrkx-gnn-tracking/configs/my_prep_med.yaml**
```
# Input/output configuration
input_dir: /data/gnn_code/trackml-particle-identification/train_all
output_dir: /data/gnn_code/heptrkx-gnn-tracking/output2
n_files: 9000

# Graph building configuration
selection:
    pt_min: 2.0 # GeV
    phi_slope_max: 0.0006
    z0_max: 150
    n_phi_sections: 4
    n_eta_sections: 2
    eta_range: [-5, 5]
```

Here you can change the input and output folders, as well as tweak some of the parameters for how the graphs are created. The output will have multiple .npz files per event, the number of files is determined by the number of sections that eta/phi get divided into, so in the above example you would get 8 files per event each covering different regions of the detector. Setting both to 1 will make a graph for the entire detector and takes a very long time to preprocess. The other parameters are cuts that can be used if desired. I am still learning all the code at the time of writing this, so its possible there are more advanced things that can be done in this preprocessing step that I am not currently aware of.

Once you have a configuration file created, you can preprocess the data as follows
```
cd /data/gnn_code/heptrkx-gnn-tracking
conda activate hgcal-env
python prepare.py configs/my_prep_med.yaml
```

Sit back, depending on the configuration this could take a very long time.

<br>

## Training the GNN

Details missing, coming soon

```
cd /data/gnn_code/hgcal_ldrd
source env.sh
python scripts/heptrx_nnconv.py -c -m=EdgeNetWithCategories -l=nll_loss -d=test_track --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW >& EdgeNetWithCategories_test.log &
tail -f EdgeNetWithCategories_test.log
```


<br>

## Plotting Results
Details missing, coming soon
```
jupyter notebook
```
**/data/gnn_code/hgcal_ldrd/notebooks/graph_generation/draw_graphs.ipynb**
