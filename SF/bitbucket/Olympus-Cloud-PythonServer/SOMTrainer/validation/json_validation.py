import jsonschema
import argparse
import json
import sys

conf_json_schema = {
    "title": "SOM Trainer configuration file",
    "type": "object",
    "properties": {
        'doIOS': {"type": "boolean"},
        'doAndroid': {"type": "boolean"},
        'pcaComponentsCount': {"type": "integer"},
        'somSize': {"type": "integer"},
        'precision': {"type": "integer"},
        'clusters': {"type": "integer"},
        'regions': {"type": "integer"},
        'iterations': {"type": "integer"},
        'maxIterations': {"type": "integer"},
        'nodeScale': {"type": "integer"},
        'dropObs': {"type": "boolean"},
        'observed_value': {"type": "number"},
        'missing_value': {"type": "number"},
        'factor': {"type": "integer"},
        'edgeLength': {"type": "number"},
        'model': {"type": "string"},
        "sigma": {"type": "number"},
        "frequency_threshold": {"type": "number"},
        "learning_rate": {"type": "number"},
        "fuzzy": {"type": "boolean"},
        "validation_type": {"type": "string"},
        "fill_low_obs_threshold": {"type": "integer"},
        "filter_data": {"type": "boolean"},
        "filter_low_observation_count": {"type": "boolean"},

        'beacons': {
            'type': "array",
            "items": {
                'label': {'type': "string"},
                'UUIDs': {"type": "array",
                          "items": {"type": "string"}
                          }
            }
        },
        'blueprint': {'type': "string"},
        'fingeprints': {
            'type': 'array',
            'items': {'type': 'string'}

        },
        'floorplan': {'type': "string"},
        'floor': {'type': "string"},
        'site': {'type': "string"},
        'owner': {'type': "string"},
        'scenario': {'type': "string"},
        'job': {'type': "string"},
        "data_path": {'type': "string"},
    },
    "required": ["doIOS", "doAndroid", "pcaComponentsCount", "somSize",
                 "precision", "clusters", "regions", "iterations", "maxIterations",
                 "dropObs", "observed_value", "missing_value", "factor", "floor", "site", "owner",
                 "data_path", "sigma", "learning_rate", "fuzzy", "frequency_threshold", "validation_type",
                 "fill_low_obs_threshold", "filter_data", "filter_low_observation_count"
                 ]

}

model_json_schema = {
    "title": "SOM Trainer Output Model",
    "type": "object",
    "properties": {
        "graph": {
            "nodes": {"type": "array",
                      "items": {
                          "neighbors": {"type": "array",
                                        "items": {
                                            "identifier": {"type": "string"},
                                            "weight": {"type": "number"}
                                        }
                                        },
                          "identifier": {"type": "string"},
                          "coordinates": {
                              "y": {"type": "integer"},
                              "x": {"type": "integer"}
                          },
                          "weight": {"type": "number"}
                      }
                      }
        },

        "jointDistributions": {"type": "array",
                               "items": {
                                   "channel": {"type": "string", "enum": ["0", "25", "26", "27"]},
                                   "direction": {"type": "string", "enum": ["N", "E", "S", "W"]},
                                   "scope": {
                                       "beacons": {"type": "array", "items": {"type": "string"}},
                                       "stat": {"type": "array", "items": {"type": "string"}},
                                   },
                                   "cov": {"type": "array", "items": {"type": "number"}},
                                   "mean": {"type": "array", "items": {"type": "number"}},

                               }},
        "som": {
            "type": "array",
            "items": {
                "channel": {"type": "string", "enum": ["0", "25", "26", "27"]},
                "direction": {"type": "string", "enum": ["N", "E", "S", "W"]},
                "normStd": {"type": "array", "items": {"type": "number"}},
                "normMean": {"type": "array", "items": {"type": "number"}},
                "somType": {"type": "string", "enum": ["som", "emsom"]},
                'pcaComponents': {'values': {"type": "array", "items": {"type": "number"}},
                                  'number': {"type": "integer"}
                                  },
                "scope": {
                    "beacons": {"type": "array", "items": {"type": "string"}},
                    "stat": {"type": "array", "items": {"type": "string"}},
                },
                'nodes': {"type": "array",
                          "items": {
                              'identifier': {"type": "string"},
                              "weights": {"type": "array", "items": {"type": "number"}},
                              "bias": {"type", "number"},
                              'nodes': {"type": "array",
                                        "items": {
                                            'identifier': {"type": "string"},
                                            'weight': {"type": "array", "items": {"type": "number"}},
                                            'means': {"type": "array", "items": {"type": "number"}},
                                            'precision': {"type": "array", "items": {"type": "number"}},
                                            'nodes': {"type": "array",
                                                      "items": {
                                                          'identifier': {"type": "string"},
                                                          'weights': {"type": "array", "items": {"type": "number"}},
                                                          'graphNodes': {"type": "array", "items": {"type": "string"}},
                                                          'threshold': {"type", "number"}
                                                      }
                                                      }
                                        }
                                        }
                          }
                          }
            }

        },

        "landmarks": {"entrances": []},

        "settings": {
            "edgeLength": {"type": "number"},
            "factor": {"type": "integer"},
            "height": {"type": "integer"},
            "width": {"type": "integer"},
            "beacons": {"type": "array", "items": {"type": "string"}}
        }

    },

    "required": ["graph", "jointDistributions", "settings", 'som']
}


def validate_json(configuration_json, json_schema):
    try:
        jsonschema.validate(configuration_json, json_schema)
        return True
    except jsonschema.ValidationError as error:
        print (error)
        return False


def json_validation():
    parser = argparse.ArgumentParser(description='Json Validator')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    group.add_argument('-c', nargs='?', dest='configuration_json', type=str,
                       help="the filepath to the configuration file")

    args = parser.parse_args()

    if args.configuration_json:
        with open(args.configuration_json, 'r') as configuration_file:
            configuration_json = json.load(configuration_file)
    else:
        configuration_json = json.loads(args.infile.readlines()[0])

    status = validate_json(configuration_json, model_json_schema)
    print("validation passed: {}".format(status))
    return status


if __name__ == '__main__':
    status = json_validation()
