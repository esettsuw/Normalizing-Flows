# Improved NF

Normalizing Flows are a method for constructing complex distributions, where both sampling and density evaluation can be efficient and exact, by transforming a simple base distribution (usually standard Gaussian or uniform) through a series of invertible or bijective transformations. 

Its performance has been enhanced  by improving three integral sections of the architecture: adding a feature extractor, using a more advanced bijective transformation, and the adjusting scoring function.

## References

Paper

- [NFAD: fixing anomaly detection using normalizing flows](https://arxiv.org/pdf/1912.09323.pdf)
- [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://proceedings.neurips.cc/paper/2020/file/ecb9fe2fbb99c31f567e9823e884dbec-Paper.pdf)

- [Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
  https://arxiv.org/abs/2008.12577)
- [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/pdf/2111.07677.pdf)



## Getting Started

You will need [Python 3.6](https://www.python.org/downloads) or later and the packages specified in _requirements.txt_.

Install packages with:

```
$ pip install -r requirements.txt
```

## Configure and Run

All configurations concerning data, model, training, visualization etc. can be made in _config.py_. !!! Do not forget to download the dataset of your choice, ex. MNIST or MVTec, and set the variables _dataset_path_ and _class_name_ in _config.py_ to run experiments on them.

To start the training, just run _main.py_.

```
$ python main.py
```

or

```
$ python3 main.py
```
