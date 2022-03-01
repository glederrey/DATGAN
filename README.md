
[![Development Status](https://img.shields.io/badge/Development%20Status-3%20--%20Alpha-green)](https://pypi.org/search/?c=Development+Status+%3A%3A+3+-+Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/DATGAN.svg)](https://pypi.python.org/pypi/DATGAN)
[![Downloads](https://pepy.tech/badge/datgan)](https://pepy.tech/project/datgan)

# DATGAN
Directed Acyclic Tabular GAN (**DATGAN**) for integrating expert knowledge in synthetic tabular data generation.
- License: [GPLv3](https://github.com/glederrey/DATGAN/blob/master/LICENSE)
- Development Status: [Alpha](https://pypi.org/search/?q=&o=&c=Development+Status+%3A%3A+3+-+Alpha)
- Homepage: https://github.com/glederrey/DATGAN

> The preprint of the article for this model will be available on arXiv by the end of February/early March.

## Overview

The **DATGAN** is a synthesizer for tabular data. It uses LSTM cells to generate synthetic data for continuous and 
categorical variable types. In addition, a Directed Acyclic Graph (DAG) can be provided to represent the structure 
between the variables and help the model to perform better.

## Requirements

The current version (v2.0.0) of the **DATGAN** works with Python 3.9 (earlier versions have not been tested) and 
Tensorflow 2. We, thus, recommend the user to set up a [virtualenv](https://virtualenv.pypa.io/en/latest/). 

## Installation

We provide a [complete installation guide](https://github.com/glederrey/DATGAN/blob/main/INSTALL.md) 
using conda and setting up a virtualenv. Please follow this guide to properly set up everything and make sure that you 
can use the DATGAN as intended.

## Testing the DATGAN

You can clone this repository and use the notebooks provided in the folder [example](https://github.com/glederrey/DATGAN/tree/main/example) 
to train the **DATGAN** and use the evaluation metrics provided in this repository.

# Data Format

## Input
The **DATGAN** uses tabular data loaded with the `pandas` library. This table must follow these rules:
- has no missing values
- has columns of types `int`, `float`, `str` or `bool`.
- each column contains data of only one type. 

**NOTE**: It is important to identify which columns are considered continuous and which are considered categorical. 
For example, columns with discrete distributions have to be defined as continuous columns.

## Output
The output of the **DATGAN** is a table of synthetic data with the same columns as the input table and as many rows as 
requested.

# Tutorial
In this short tutorial we will guide you through a series of steps that will help you getting
started with the most basic usage of **DATGAN** in order to generate samples from a given dataset.

**NOTE**: The following examples are also covered in a [Jupyter](https://jupyter.org/) notebook,
which you can execute by running the following commands inside your *virtualenv*:

```
pip install jupyter
jupyter notebook example/training.ipynb
```

### 1. Load the data

The first step is to load the data wich we will use to fit the **DATGAN**. In the example, we provide a demo dataset, 
the **CMAP** dataset. You can load it using `pandas`. We also need to define which columns are considered continuous. 
For this, we simply define a list of `str` with the name of the variables that we consider as continuous.
```python
import pandas as pd

df = pd.read_csv('./data/CMAP.csv', index_col=False)

continuous_columns = ["distance", "age", "departure_time"]
```

### 2. Create a DAG

The second steps consists in creating the DAG for the DATGAN. The DAG is created using the library `networkx` from 
Python. *If you just want to test the model without any specific DAG, we recommend you to use a linear DAG.*

Example of a DAG for the CMAP dataset:
```python
import networkx as nx

graph = nx.DiGraph()
graph.add_edges_from([
    ("age", "license"),
    ("age", "education_level"),
    ("gender", "work_status"),
    ("education_level", "work_status"),
    ("education_level", "hh_income"),
    ("work_status", "hh_income"),
    ("hh_income", "hh_descr"),
    ("hh_income", "hh_size"),
    ("hh_size", "hh_vehicles"),
    ("hh_size", "hh_bikes"),
    ("work_status", "trip_purpose"),
    ("trip_purpose", "departure_time"),
    ("trip_purpose", "distance"),
    ("travel_dow", "choice"),
    ("distance", "choice"),
    ("departure_time", "choice"),
    ("hh_vehicles", "choice"),
    ("hh_bikes", "choice"),
    ("license", "choice"),
    ("education_level", "hh_size"),
    ("work_status", "hh_descr"),
    ("work_status", "hh_size"),
    ("hh_income", "hh_bikes"),
    ("hh_income", "hh_vehicles"),
    ("trip_purpose", "choice")
])
```

If you do not have any idea how to create your DAG, it is possible to not provide any DAG to the model. However, in 
this case, the model will define a _linear_ DAG, _i.e._ each variable in the dataset is linked to the next one following
the order of the columns. It can be useful to quickly test the model. However, it will reduce the performance of the 
model as shown in the article.

### 3. Create a DATGAN instance

The next step is to import **DATGAN** and create an instance of the model. There are no required parameters for the 
model. However, we advise you to set up the basic parameters such as the output folder (`output`), batch size 
(`batch_size`), and verbose level (`verbose`).

```python
output_folder = './output/'
batch_size = 558

from datgan import DATGAN

datgan = DATGAN(output=output_folder, batch_size=batch_size, verbose=1)
```

**NOTE**: Setting up a suitable batch size is really important. A batch size too big will make the model crash due to 
memory error while one that is too small will make the model slower to train. Trials and errors are required 
depending on your hardware. In addition, it is good to find a batch size such that `len(df) % batch_size` is as small 
as possible since the last batch of data is dropped if it is smaller than the batch size. 

### 4. Preprocess the data (optional)

The fourth step consists in preprocessing the data. This step is optional since it is automatically done in the next 
step if skipped right now. We propose to do the preprocessing in advance because it usually takes a bit of time. And if 
you want to try multiple parameters with the **DATGAN**, you do not have to preprocess the data every time. Therefore, 
it is possible to do it before fitting the model and saving it somewhere. 

```python
datgan.preprocess(df, continuous_columns, preprocessed_data_path='./encoded_data')
```

### 5. Fit the model
Once you have a **DATGAN** instance, you can call the method `fit` and passing the following parameters:
- `data`: the original DataFrame
- `graph`: the `networkx` DAG
- `continuous_columns`: the list of continuous columns
- `preprocessed_data_path`: the path to the preprocessed data if done in Step 4.
```python
datgan.fit(df, graph, continuous_columns, preprocessed_data_path='./encoded_data')
```

### 6. Sample new data
Once the model has been fitted, you can generate new synthetic data by calling the function `sample`. You have to 
provide the desired number of samples.
```python
samples = datgan.sample(len(df))
samples.age = np.round(samples.age)
samples.to_csv('./data/CMAP_synthetic.csv', index=False)
```
In this case, the column `age` is a discrete distribution. The **DATGAN** cannot provide such data type for the moment, 
we, thus, advise you to treat such column as continuous and, then, round the values.

### 7. Save and load a model

In the steps above we saw that the fitting process can take a lot of time, so we probably would
like to avoid having to fit every we want to generate samples. We advise the use to save checkpoints of the model while
it is training. However, if you do not want to do that, the model will always save the latest checkpoint once it has
finished training. You can, thus, load it at any time afterwards.

In order to load the model, you can simply call the function `load` with the parameters used while fitting the model.
In order to save memory, we only save the parameters of the Generator and Discriminator. Therefore, more information is
required to load the model.

```python
new_datgan = datgan.load(df, graph, continuous_columns, preprocessed_data_path='./encoded_data')
```

At this point we can use this model instance to generate more samples.

# Model parameters

If you want to change the default behavior of `DATGAN`, such as using different batch size or the total number of 
epochs, you can do so by passing different arguments when creating the **DATGAN** instance.

## Loading the model

| Name               |  Type   |   Default    | Explanation                                                                                                                                                                                                                  |
|:-------------------|:-------:|:------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `loss_function`    |  `str`  |    `None`    | Name of the loss function to be used. If not specified, it will choose between `'WGAN'` and `'WGGP'` depending on the ratio of continuous and categorical columns. Only accepts the values `'SGAN'`, `'WGAN'`, and `'WGGP'`. |
| `label_smoothing`  |  `str`  |    `'TS'`    | Type of label smoothing. Only accepts the values `'TS'`, `'OS'`, and `'NO'`.                                                                                                                                                 |
| `output`           |  `str`  | `'./output'` | Path to store the model and its artifacts.                                                                                                                                                                                   |
| `gpu`              |  `int`  |    `None`    | Model will automatically try to use GPU if tensorflow can use CUDA. However, this parameter allows you to choose which GPU you want to use.                                                                                  |
| `num_epochs`       |  `int`  |    `100`     | Number of epochs to use during training.                                                                                                                                                                                     |
| `batch_size`       |  `int`  |    `500`     | Size of the batch to feed the model at each step.                                                                                                                                                                            |
| `save_checkpoints` | `bool`  |    `True`    | Whether to store checkpoints of the model after each training epoch.                                                                                                                                                         |
| `restore_session`  | `bool`  |    `True`    | Whether continue training from the last checkpoint.                                                                                                                                                                          |
| `learning_rate`    | `float` |    `None`    | Learning rate. If set to None, the value will be set according to the chosen loss function.                                                                                                                                  |
| `g_period`         |  `int`  |    `None`    | Every `g_period` steps, train the generator once. (Used to train the discriminator more than the generator) By default, it will choose values according the chosen loss function.                                            |
| `l2_reg`           | `bool`  |    `None`    | Tell the model to use L2 regularization while training both NNs. By default, it only applies the L2 regularization when using the SGAN loss function.                                                                        |
| `z_dim`            |  `int`  |    `200`     | Dimension of the noise vector used as an input to the generator.                                                                                                                                                             |
| `num_gen_rnn`      |  `int`  |    `100`     | Size of the hidden units in the LSTM cell.                                                                                                                                                                                   |
| `num_gen_hidden`   |  `int`  |     `50`     | Size of the hidden layer used on the output of the generator to act as a convolution.                                                                                                                                        |
| `num_dis_layers`   |  `int`  |     `1`      | Number of layers for the discriminator.                                                                                                                                                                                      |
| `num_dis_hidden`   |  `int`  |    `100`     | Size of the hidden layers in the discriminator.                                                                                                                                                                              |
| `noise`            | `float` |    `0.2`     | Upper bound to the gaussian noise added to with the label smoothing. (only used if `label_smoothing` is set to `'TS'` or `'OS'`)                                                                                             |
| `verbose`          |  `int`  |     `1`      | Level of verbose. 0 means no print, 1 means that some details will be printed, 2 is mostly used for debugging purpose.                                                                                                       |


## Sampling synthetic data

When sampling the synthetic data (`DATGAN.sample`), you can choose between multiple sampling strategies. 

| Name       | Type  | Default | Explanation                                                                                     |
|:-----------|:-----:|:-------:|:------------------------------------------------------------------------------------------------|
| `sampling` | `str` |  `SS`   | Type of sampling to use. Only accepts the following values: `'SS'`, `'SA'`, `'AS'`, and `'AA'`. |


`S` means we are using simulation to sample the data, `A` means that we are using argmax. The first letter corresponds 
to continuous variables and the second to categorical variables. Therefore, `SA` means we're using simulation for 
continuous variables and argmax for categorical variables.

# Tips and tricks

While the **DATGAN** model will automatically choose the parameters of the model if none are provided, we highly 
recommend the user to "play" with them. The most important ones are the following:
- `loss_function` Generally, the `WGAN` loss function works better on datasets with more categorical columns than 
continuous. It is the contrary for the `WGGP` loss. The `SGAN` loss seems to perform a bit less good than the other two
in the specific cases. However, it seems to perform ok in any cases. 
- `g_period` This parameter is especially important when using either the `WGAN` or the `WGGP` loss. Generally, the 
latter requires a lot more training of the discriminator than the previous. However, it might be interesting to test 
different values to see which one leads to the best results.
- `l2_reg` The L2 regularization is "mandatory" for the `SGAN` loss and "forbidden" for the `WGAN` loss. The user can 
test these other configurations, but it will lead to worse results. However, for the `WGGP` loss, there are no specific
rules whether to apply it or not. Therefore, it might be interesting to test this parameter as well.
- `learning_rate` It has been fixed depending on the loss function. However, as for any optimization problem, playing 
with the learning rate to find the optimal value is always important.

# Acknowledgements

We would like to thank the authors of the **TGAN** article, Lei Xu and Kalyan Veeramachaneni, as well as all the 
contributors of the TGAN Github repository. This model has greatly inspired the ideas behind the **DATGAN** and we have 
used their code as a starting point to write our model. 

# Citing DATGAN

If you use DATGAN or its evaluation metrics, please cite the following work:

> Gael Lederrey, Tim Hillel, Michel Bierlaire. 2022. *DATGAN: Integrating expert knowledge into deep learning for 
> synthetic tabular data*. ArXiv preprint

**FULL CITATION COMING SOON!**

The original code for this article can be found in this Github repository: https://github.com/glederrey/SynthPop.
