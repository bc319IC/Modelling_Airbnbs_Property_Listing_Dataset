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
Run the appropraite files in the following section regarding their descriptions.

## File Structure

### tabular_data.py <a id="tabular_data.py"></a>
Contains functions to clean the tabular data and load the features and columns as a tuple. Run this file to download the cleaned version of the CSV file.

### modelling.py <a id="modelling.py"></a>
Run this file to generate the best models after specifing the type of problem (classification or regression), the model (neural network or not), and the label.

#### regression.py <a id="regression.py"></a>
Contains the code for generating regression models.

#### classification.py <a id="classification.py"></a>
Contains the code for generating classification models.

#### regression_nn.py <a id="regression_nn.py"></a>
Contains the code for generating regression neural networks.

#### classification_nn.py <a id="classification_nn.py"></a>
Contains the code for generating classification neural networks.

## License
This project is licensed under the terms of the MIT license. (tbd)