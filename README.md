# Company Bankruptcy Prediction
A model to predict whether a company will file for bankruptcy

## Model features
- Original number of features: 95
- Reduced number of features before clustering: TBD
- Reduced features in stack models: each ensemble may use different number of features

## Model target
- Class = 1 (File for bankrupcy)
- Class = 0 (Does not file for bankrupcy)

## Project setup 
### Clone this repository to your machine
```
git clone https://github.com/jefreybulla/company-bankruptcy-prediction.git
```

### Create Python environment
- Install python 3 if needed.
- Create environment with 
```
python3 -m venv company-bankruptcy-prediction-env
```

### Run in VS code
- Open project in VS code
- In the UI, pick the kernel: company-bankruptcy-prediction

### Run in Jupyter
- Use Python environment with
```
source company-bankruptcy-prediction-env/bin/activate
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

## Using joblib
`joblib` is used for data processing. You will need the latest version of joblib. As of April 2025 the latest version in 1.5.3. You can check your joblib version with
```
pip list 
```
Find joblib in the list and check your version. If needed install the latest version with 
```
pip install joblib
```
