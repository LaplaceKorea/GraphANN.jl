# Python(-ish) bindings to GraphANN

This is primarily meant as a compatibility layer between GraphANN and the [`big-ann-benchmarks`](https://github.com/harsha-simhadri/big-ann-benchmarks) project.

## Installation

Version of Python are a headache to manage.
To develop locally, clone the `big-ann-benchmarks` repo and copy the local file `/build/graphann.py` into `big-ann-benchmarks/benchmark/algorithms`
```
export BIGANN=<path/to/bigann>
export PYANN_ROOT=<path/to/pyann>
git clone https://github.com/harsha-simhadri/big-ann-benchmarks $BIGANN
cp $PYANN_ROOT/build/graphann.py $BIGANN/benchmark/algorithms/graphann.py
```
Next, follow the steps outlined [here](https://github.com/pyenv/pyenv-installer) to install `pyenv` and configure `pyenv` to use Python 3.6.15 (the highest version of Python 3.6 which is the default for Ubuntu 18.04 - the OS version being used in the Docker containers).
The process will look something like
```
cd $BIGANN
pyenv local 3.6.15
pyenv global 3.6.15
pip3 install -r requirements.txt
pip3 install --user julia
python -c "import julia; julia.install()"
```
The GraphANN Python wrapper can then be imported from Python using
```python
import benchmark.algorithms.graphann as gr
```
