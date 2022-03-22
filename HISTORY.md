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