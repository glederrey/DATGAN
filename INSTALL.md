# Installation

Here are the steps to setup a virtual environment with conda, install the **DATGAN** package and use it.

# Requirements:

Make sure the following software are installed on your machine.

- Any Python version
- Conda
- Jupyter Notebook
- CUDA Toolkit 11.6 (Download it [here](https://developer.nvidia.com/cuda-downloads).)
- cuDNN (Download it [here](https://developer.nvidia.com/cudnn))

# Steps to install the DATGAN library

We highly recommend creating a virtual environment to use the **DATGAN** library!

## Setting up the virtual environment

1. We can directly use conda to setup the virtual environment using the following command: (`ENV_NAME` corresponds to 
   the name of the environment you want to give)

```python
conda create -n ENV_NAME python=3.9
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

## Installing the DATGAN library

You can now install the **DATGAN** library using the following command:

```python
pip install datgan
```

The **DATGAN** library will install all the requirements for the library to function correctly. However, it might be a 
good idea to install Tensorflow 2 manually and test if it find your GPU. (Some issues can happen at this step.)

# Testing the DATGAN library

You can now clone the repository and test the **DATGAN** using the files in the folder `example`. 

Enjoy! 🥳