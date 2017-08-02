"""
json_schema to validated all somtrainer configuration json

"""
import jsonschema
import argparse
import json
import sys

json_schema = {
    "title": "SOMTRainerConfiguration",
    "type": "object",
    "properties": {
        "ios": {"type": "boolean",
                "description": "Train iOS",
                },
        "android": {"type": "boolean",
                    "description": "Train android",
                    },
        "components": {"type": "number",
                       "description": "Number of variance percentage of components for PCA",
                       },
        "use_max": {"type": "boolean",
                    "description": "Bool True if max is a feature",
                    },
        "use_mad": {"type": "boolean",
                    "description": "Bool True if mad is a feature"
                    },
        "use_median": {"type": "boolean",
                       "description": "Bool True if median is a feature"
                       },
        "data_path": {"type": "string",
                      "description": "root path where data resides"
                      },
        "site": {"type": "string",
                 "description": "site path where data resides"
                 },
        "location": {"type": "string",
                     "description": "location path where data resides"
                     },
        "factor": {"type": "integer",
                   "description": "factor of pixels per node in plots"
                   },
        "nodes": {"type": "integer",
                  "description": "number of nodes per dimension in SOM"
                  },
        "precision": {"type": "integer",
                      "description": "number of decimal places to keep in json weights"
                      },
        "cluster": {"type": "integer",
                    "description": "number of clusters to use in EMSOM"

                    },
        "iterations": {"type": "integer",
                       "description": "number of iterations to run training"
                       },

        "max_iterations": {"type": "integer",
                           "description": "number of max iterations to run training"
                           },

        "node_scale": {"type": "integer",
                       "description": "scaling factor for emsom sublayer node dimensions"
                       },
        "edge_length":{"type": "number",
                       "description": "the edge length per node"
        }
    },

    "required": ["ios", "android","components","use_mad", "use_max","use_median",
                 "data_path", "site", "location", "nodes", "precision",
                 "cluster", "iterations", "max_iterations", "node_scale"
                 ]
}


def validate_json(configuration_json):
    try:
        jsonschema.validate(configuration_json, json_schema)
        return True
    except jsonschema.ValidationError as error:
        print(error)
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

    status = validate_json(configuration_json)
    print("validation passed: {}".format(status))
    return status

if __name__ == '__main__':
    status = json_validation()
