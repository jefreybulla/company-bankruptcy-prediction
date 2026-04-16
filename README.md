# Company Bankruptcy Prediction
A model to predict whether a company will file for bankruptcy

## Model features
- Oroginal features: 95 features
- Reduced features before clustering: TBD
- Reduced features in stack models: each ensemble may use different number of features

## Model target
- Class = 1 (File for bankrupcy)
- Class = 0 (Does not file for bankrupcy)

## Project setup 
## Create Python environment
- Install python 3 if needed.
- Create environment with 
```
python3 -m venv insider-trading-signal-env
```

### Run in VS code
- Open project in VS code
- In the UI, pick the kernel: insider-trading-signal

### Run in Jupyter
- Use Python environment with
```
source insider-trading-signal-env/bin/activate
```
- Update pip
```
pip install --upgrade pip
```
- Open Jupyter notebooks
- Exit Python environment with
```
deactivate
```

## Using tensorflow
As of April 2025 Tensorflow does not support Python 3.13. We need to use Python 3.11. 
- Install Python 3.11
```
brew install python@3.11
```
- Confirm installation
```
python3.11 --version
```
- Create environment for Tensorflow
```
python3.11 -m venv tf-env
source tf-env/bin/activate
pip install --upgrade pip
pip install tensorflow   # on Intel Mac
# or: pip install tensorflow-macos   # on Apple silicon