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
Next, follow the steps outlined [here](https://github.com/pyenv/pyenv-installer) to install `pyenv` and configure `pyenv` to use Python 3.8.
Before proceeding with the below steps - make sure that the Julia binary is located on the system path.
For example, if you downloaded Julia-1.6.3, then you will want to do:
```
export PATH=<path/to/julia>/julia-1.6.3/bin:$PATH
```
Setting up pyenv looks like this:
```
cd $BIGANN  
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8.12
pyenv local 3.8.12
pyenv global 3.8.12
pip3 install -r requirements.txt
pip3 install --user julia
python -c "import julia; julia.install()"
```
Afterwards, add the following to your `.bashrc/.zshrc` file:
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

The GraphANN Python wrapper can then be imported from Python using
```python
import benchmark.algorithms.graphann as gr
```
