Feature: Load CSV data and process it saving the results to pickle files using different features

  @features
  Scenario Outline: process fingerprint data with features median, max, frequency
    Given the configuration json file with channel <channel> and features median max frequency
    When the raw fingerprints are loaded from csv
    Then Data is processed with <component> pcaComponentsCount
    And Processed Data is saved to pickle
    And Processed data is loaded from pickle


  Examples: pcaComponentsCount channels combination
      | channel | component|
      | 0| 0 |
      | 25 | 0 |
      | 26 | 0 |
      | 27 | 0 |
      | 0| 5|
      | 25 | 5 |
      | 26 | 5 |
      | 27 | 5 |


#  @validationfeatures
#  Scenario Outline:  process validation data with features median, max, frequency
#    Given the configuration json file with channel <channel> and features median max frequency
#    When the raw fingerprints are loaded from csv
#    Then Rolled Data is processed with window size: 5 and step: 2
#    And Rolled Data is saved to pickle
#    And Data is processed with <component> pcaComponentsCount
#    And Rolled Data is loaded from pickle
#    And Validation Data is created from Rolled Data
#    And Validation Data is saved to pickle
#    And validation data is loaded from pickle
#
#  Examples: pcaComponentsCount channels combination
#      | channel | component|
#      | 0| 0 |
#      | 25 | 0 |
#      | 26 | 0 |
#      | 27 | 0 |
#      | 0| 5|
#      | 25 | 5 |
#      | 26 | 5 |
#      | 27 | 5 |
