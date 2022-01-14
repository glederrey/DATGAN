# Installation

Here are the steps to setup a virtual environment with conda, install the **DATGAN** package and use it. 

>⚠️ The current version of the **DATGAN** only works with Python 3.7 and inside a Jupyter Notebook. This issue will be 
> fixed in later versions.

# Requirements:

Make sure the following software are installed on your machine.

- Any Python version
- Conda
- Jupyter Notebook

# Steps to install the DATGAN library

We highly recommend creating a virtual environment to use the **DATGAN** library!

## Setting up the virtual environment

1. We can directly use conda to setup the virtual environment using the following command: (`ENV_NAME` corresponds to 
   the name of the environment you want to give)

```python
conda create -n ENV_NAME python=3.7
```

1. We can activate the environment using the command below. On Windows, you might have issues with your current shell. 
   Therefore, we recommend using the Anaconda Prompt on Windows. The terminal should work on Linux and MacOS.

```python
conda activate ENV_NAME
```

2. We, now, need to activate the virtual environment to make it work with Jupyter notebook. Use the following command:

```python
pip install --user ipykernel
```

3. We need to **manually add the kernel**. Use the following command:

```python
python -m ipykernel install --user --name=ENV_NAME
```

4. You can start Jupyter notebook. If everything worked correctly, you should see the new environment when clicking on **New**.

## Installing tensorflow

> ⚠️ **Warning**: You need to stay in the same environment while installing tensorflow and the **DATGAN** library. If 
> you’re not sure, open Anaconda Prompt or a terminal and reactivate the environment as under point 2.

You need to install tensorflow. We recommend doing it before installing the library. If you have a GPU available, 
install tensorflow using the following command:

```python
pip install tensorflow-gpu==1.15
```

You can test that the tensorflow was correctly installed using the following command (it should return `True`):

```python
tensorflow.test.is_gpu_available()
```

If you do not have a GPU available, you can install tensorflow normally. However, the model will be very slow to train. 
To install the tensorflow version without GPU, use the following command:

```python
pip install tensorflow==1.15
```

## Installing the DATGAN library

You can now install the **DATGAN** library using the following command:

```python
pip install datgan
```

# Testing the DATGAN library

You can now clone the repository and test the **DATGAN** using the files in the folder `example`. 

Enjoy! 🥳