Feature: Load CSV data and create features for all different options

  @loader
  Scenario Outline: Load CSV Raw fingerprint data
    Given the default test configuration json file with channel <channel>
    When the raw fingerprints are loaded from csv
    Then the features are <features>
    And the data are filtered <filtered>
    And the low observations are dropped <drop_low>
    And the missing observation value is -120
    And the features tuple is created


  Examples: pcaComponentsCount channels combination
      | channel | features| filtered | drop_low|
      | 0 | median,max | False | False|
      | 25 | median,max| False |False |
      | 26 | median,max| False | False |
      | 27 | median,max| False | False |
      | 0 | median,max,frequency| True | False|
      | 25 | median,max,frequency| True |False |
      | 26 | median,max,frequency| True | False |
      | 27 | median,max,frequency| True | False |
      | 0 | median,max| False | True|
      | 25 | median,max| False |True |
      | 26 | median,max| False | True |
      | 27 | median,max| False | True |
      | 0 | median,max,frequency| False | True|
      | 25 | median,max,frequency| False |True |
      | 26 | median,max,frequency| False | True |
      | 27 | median,max,frequency| False | True |
