# Modelling Airbnbs Property Listing Dataset
Development of an end to end pipeline for creating machine learning models for both regression and 
classification, including neural networks to predict on the airbnb dataset.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li>
      <a href="#File-Structure">File Structure</a>
      <ul>
        <li><a href="#tabular_data.py">tabular_data.py</a></li>
        <li>
          <a href="#modelling.py">modelling.py</a>
          <ul>
            <li><a href="#regression.py">regression.py</a></li>
            <li><a href="#classification.py">classification.py</a></li>
            <li><a href="#regression_nn.py">regression_nn.py</a></li>
            <li><a href="#classification_nn.py">classification_nn.py</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#License">License</a></li>
  </ol>
</details>

## Installation
Clone for local access.
```sh
git clone https://github.com/bc319IC/modelling-airbnbs-property-listing-dataset-338.git
```

## Usage
Running the python file `modelling.py` will generate the best regression/classification models depending on what parameters have been set for the given CSV file and label column.

## File Structure

### tabular_data.py <a id="tabular_data.py"></a>
Contains functions to clean the tabular data and load the features and labels columns as a tuple. Running this file downloads the cleaned version of the CSV file where the cleaning functions are particular to the CSV file used in this project.

### modelling.py <a id="modelling.py"></a>
This is the final combined script of the below scripts to create a complete pipeline of generating machine learning models with their metrics.

#### regression.py <a id="regression.py"></a>
Contains the code for generating regression models and their metrics.

#### classification.py <a id="classification.py"></a>
Contains the code for generating classification models and their metrics.

#### regression_nn.py <a id="regression_nn.py"></a>
Contains the code for generating regression neural network models and their metrics.

#### classification_nn.py <a id="classification_nn.py"></a>
Contains the code for generating classification neural network models and their metrics.

## License
This project is licensed under the terms of the MIT license.