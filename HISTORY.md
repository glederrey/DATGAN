# History

## 1.0

* First release on PyPI.

## 1.1

* Completed the README with all the details how to use DATGAN.

## 1.1.1 

* README updated and moved installation file to Github.

## 2.0.0

* Rewritten the code to work with tensorflow 2. 
* Code has been tested against version 1.1.1 to make sure the generated datasets 
provide the same results on the assessments results.
* Added some more "customization" options for the training of the models.

## 2.0.1

* Add a new way to define the data types to better control continuous columns

## 2.1.0

* Add two types of conditionality to the DATGAN (rejection sampling and conditional inputs)

## 2.1.1

* Add a way to enforce the bounds instead of discarding values outside of bounds. (Useful for mixed
distributions with peak of values close to one of the bounds.)

## 2.1.2

* Add a function to help when building the DAG.

## 2.1.3

* Library tested on Python 3.7 => removed requirements for Python 3.9 only. Now accepts Python >=3.7

## 2.1.4

* Fixed issue with the computation of the KL divergence

## 2.1.5

* Updated the README based on comments given by users.

## 2.1.6

* Add possibility to ignore some columns while assessing the results
* Fixed an issue with the DAG when using conditional inputs
    * It was removing the edges between variables in the conditional inputs, thus leading to worse results since 
    some of the variables in the conditional inputs were not taken into account in the attention vectors.
* Revert to using the direct output of the LSTM cells for the attention vectors.

## 2.1.7

* Fixed an issue when transforming the DAG according to the conditional inputs.

## 2.1.8

* Added the option to sample data from the DATGAN without randomizing the conditional inputs.
