# 1. Create a clean env with everything we need
conda create -n perceptron-demo python=3.11 numpy matplotlib jupyterlab ipykernel -y

# 2. Activate it
conda activate perceptron-demo

# 3. Register this environment as a Jupyter kernel
python -m ipykernel install --user --name perceptron-demo --display-name "Perceptron (conda)"

