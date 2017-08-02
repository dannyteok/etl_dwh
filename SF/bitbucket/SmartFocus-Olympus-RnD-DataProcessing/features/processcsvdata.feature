Feature: Load CSV data and process it saving the results to pickle files

  @features
  Scenario Outline: Standard Directory process fingerprint data
    Given the default test configuration json file with channel <channel>
    When the raw fingerprints are loaded from csv
    Then the features are ["median", "max"]
    And Data is processed with <component> pcaComponentsCount
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

#  @validation
#  Scenario Outline:  Standard Directory process validation data
#    Given the default test configuration json file with channel <channel>
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

  @filterfeatures
  Scenario Outline: Standard Directory process fingerprint data filtered
    Given the default test configuration json file with channel <channel>
    When the raw fingerpints are loaded from csv with a few filtered beacons
    Then Raw Data is saved to pickle
    And Data is processed with <component> pcaComponentsCount
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

#  @filtervalidation
#  Scenario Outline:  Standard Directory process validation data filtered
#    Given the default test configuration json file with channel <channel>
#    When the raw fingerpints are loaded from csv with a few filtered beacons
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
#
#
#  @brighton
#  Scenario Outline: Brighton file process fingerprint data
#    Given the Brighton configuration json file with channel <channel>
#    When the raw fingerprints are loaded from csv
#    Then Raw Data is saved to pickle
#    And Data is processed with <component> pcaComponentsCount
#    And Processed Data is saved to pickle
#    And Processed data is loaded from pickle
#
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
#
#
#  @brightonvalidation
#  Scenario Outline:  Standard Directory process validation data
#    Given the Brighton configuration json file with channel <channel>
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
