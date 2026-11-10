# MNIST Neural Network from Scratch (NumPy)

A fully hand-written neural network for classifying handwritten digits from the MNIST dataset, implemented using NumPy.

## Dataset

- **Source:** MNIST (via `tf.keras.datasets.mnist.load_data()` - used *only* for downloading)
- **Format:** Converted to two CSV files by `fetch_mnist.py`
  - `data/mnist_train.csv`
  - `data/mnist_test.csv`

The download step uses TensorFlow, but the **model itself is 100% NumPy**. If you prefer zero TensorFlow dependency, you can swap in any MNIST CSV source.

## Installation/Usage

- MNIST dataset is fetched online via an API (tensorflow) hence an internet connection is required to access the dataset for training and testing.

### 1. Clone the repo
```bash
git clone https://github.com/einye/mnist-from-scratch-numpy.git
cd mnist-from-scratch-numpy

# (Recommended) Create a virtual environment
python3.12 -m venv venv         # Tested on Python 3.12
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download & prepare MNIST
python fetch_mnist.py
This creates data/mnist_train.csv and data/mnist_test.csv.

# Train and evaluate
python main.py
```

## Acknowledgements

- Yann LeCun et al. for the MNIST dataset and the classic architecture notes

- The endless blog posts and papers, notably ["Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition" by D. Ciresan et al](https://arxiv.org/pdf/1003.0358), that describes Multi-layer perceptrons (MLP) the network architecture is based on.
