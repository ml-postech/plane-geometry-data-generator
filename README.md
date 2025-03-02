# Benchmark and Data Generator for Plane Geometry Diagram Understanding

This is the code for generating visual geometric premises recognition benchmark and GeoCLIP-style data used for evaluating and constructing the vision encoder of [GeoDANO](https://arxiv.org/pdf/2502.11360). The code is built on the implementation of [AlphaGeometry](https://github.com/google-deepmind/alphageometry).

## Installation

Installation is done in a virtual environment:

```
virtualenv -p python3 .
source ./bin/activate
pip install --require-hashes -r requirements.txt
```

Install `meliad` separately as it is not
registered with `pip`:

```
MELIAD_PATH=meliad_lib/meliad
mkdir -p $MELIAD_PATH
git clone https://github.com/google-research/meliad $MELIAD_PATH
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH
```

## Benchmark Generation

Run the following script to generate the benchmark. The default directory is `./data`:

```
bash benchmark_generation.sh
```

## GeoCLIP-style Data Generation

Run the following scrip to generate the GeoCLIP-style data used to train the vision encoder:

```
python clip_generator.py --n_problems 200000 --image_folder ./data/clip/images --out_file ./data/clip/problems.jsonl --n_workers 50
```
