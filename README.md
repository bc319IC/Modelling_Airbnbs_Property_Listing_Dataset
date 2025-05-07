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
    <li>
      <a href="#Results">Results</a>
      <ul>
        <li>
          <a href="#Regression">Regression</a>
          <ul>
            <li><a href="#SGD-Regressor">SGD Regressor</a></li>
            <li><a href="#Decision-Tree-Regressor">Decision Tree Regressor</a></li>
            <li><a href="#Random-Forest-Regressor">Random Forest Regressor</a></li>
            <li><a href="#Gradient-Boosting-Regressor">Gradient Boosting Regressor</a></li>
            <li><a href="#Regression-Neural-Network">Regression Neural Network</a></li>
          </ul>  
        </li>
        <li>
          <a href="#Classification">Regression</a>
          <ul>
            <li><a href="#Logistic-Regression">Logistic Regression</a></li>
            <li><a href="#Decision-Tree-Classifier">Decision Tree Classifier</a></li>
            <li><a href="#Random-Forest-Classifier">Random Forest Classifier</a></li>
            <li><a href="#Gradient-Boosting-Classifier">Gradient Boosting Classifier</a></li>
            <li><a href="#Classification-Neural-Network">Classification Neural Network</a></li>
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

## Results

### Regression
Best performing models on predicting the price per night and their metrics.

| Model | Train RMSE | Train R2 | Validation RMSE | Validation R2 | Test RMSE | Test R2 |
| --- | --- | --- | --- | --- | --- | --- |
| SGD Regressor | 101.22058912625033 | 0.4075008874418633 | 596.3715198944476 | -17.09545021601175 | 636.5735006698967 | -47.45800079778052 |
| Decision Tree Regressor | 97.26824956102288 | 0.45286791482116817 | 107.20875578632834 | 0.4152151484696317 | 69.86114850326632 | 0.4163670224399525 |
| Random Forest Regressor | 74.0856384720796 | 0.6825918755587166 | 99.3242267292986 | 0.4980667250531242 | 62.690647696565925 | 0.5300259325741634 |
| Gradient Boosting Regressor | 56.185429735476255 | 0.8174434596845955 | 100.77313547388734 | 0.48331584269532035 | 66.46722103445607 | 0.47169657760322514 |
| Regression Nerual Network | 101.55003513631439 | 0.40363775788496004 | 111.38395842558414 | 0.36877978563824 | 58.22297564970172 | 0.5946247998155073 |

#### SGD Regressor
| Best Hyperparameters | alpha : 0.1 | eta0: 0.01 | max_iter: 20000 | penalty: elasticnet |
| --- | --- | --- | --- | --- |

#### Decision Tree Regressor
| Best Hyperparameters | max_depth: 10 | min_samples_leaf: 4 | min_samples_split: 40 |
| --- | --- | --- | --- |

#### Random Forest Regressor
| Best Hyperparameters | max_depth: 20 | min_samples_leaf: 4 | min_samples_split: 2 | n_estimators: 50 |
| --- | --- | --- | --- | --- |

#### Gradient Boosting Regressor
| Best Hyperparameters | learning_rate: 0.01 | max_depth: 7 | n_estimators: 200 | subsample: 0.6 |
| --- | --- | --- | --- | --- |

#### Regression Neural Network
| Best Hyperparameters | learning_rate: 0.01 | hidden_layer_width: 256 | depth: 4 | optimiser: adam |
| --- | --- | --- | --- | --- |

### Classification
Best performing models on predicting the property category and their metrics.

| Model | Train Accuracy | Train F1 | Validation Accuracy | Validation F1 | Test Accuracy | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.4343891402714932 | 0.4225976975626674 | 0.43373493975903615 | 0.4199081200601588 | 0.3373493975903614 | 0.3297852635029916 |
| Decision Tree Classifier | 0.5158371040723982 | 0.5133402814259708 | 0.40963855421686746 | 0.4134847677856924 | 0.24096385542168675 | 0.23767736970452957 |
| Random Forest Classifier | 0.6847662141779789 | 0.6835612899053704 | 0.40963855421686746 | 0.39299031876297463 | 0.30120481927710846 | 0.29047523427041494 |
| Gradient Boosting Classifier | 0.5203619909502263 | 0.5080691092751083 | 0.40963855421686746 | 0.3908154356556662 | 0.30120481927710846 | 0.28682402678299546 |
| Classification Nerual Network | 0.3273001508295626 | 0.23928847499883732 | 0.3614457831325301 | 0.2703406794881389 | 0.20481927710843373 | 0.11600412344565426 |

#### Logistic Regression
| Best Hyperparameters | C : 10 | max_iter: 10000 | solver: liblinear |
| --- | --- | --- | --- |

#### Decision Tree Classifier
| Best Hyperparameters | max_depth: nul | min_samples_leaf: 4 | min_samples_split: 40 |
| --- | --- | --- | --- |

#### Random Forest Classifier
| Best Hyperparameters | max_depth: 10 | min_samples_leaf: 2 | min_samples_split: 20 | n_estimators: 50 | 
| --- | --- | --- | --- | --- |

#### Gradient Boosting Classifier
| Best Hyperparameters | learning_rate: 0.1 | max_depth: 1 | n_estimators: 100 | subsample: 0.6 |
| --- | --- | --- | --- | --- |

#### Classification Neural Network
| Best Hyperparameters | learning_rate: 0.001 | hidden_layer_width: 256 | depth: 2 | optimiser: adam |
| --- | --- | --- | --- | --- |

## License
This project is licensed under the terms of the MIT license.