import os
import hashlib
from six import iteritems
import glob
import re
import numpy as np
from collections import OrderedDict
"""
Configurations creates a class with all the configuration variables to use in the parsers

It should provide validation and also make it easy to pass the same variables around

"""


class ProcessingConfiguration(object):

    @classmethod
    def create_configuration(cls, dictionary):

        return cls(
            owner=dictionary['owner'],
            site=dictionary['site'],
            floor=dictionary['floor'],
            data_path=dictionary['data_path'],
            pcaComponentsCount=dictionary['pcaComponentsCount'],
            filter_beacons=dictionary.get('filter_beacons',None),
            observed_value=dictionary['observed_value'],
            missing_value=dictionary['missing_value'],
            features=dictionary['features'],
            beacons=dictionary['beacons'],
            dropObs=dictionary['dropObs'],
            channel=dictionary.get("channel", "0"),
            window_size=dictionary['window_size'],
            window_step=dictionary['window_step'],
            log_data=dictionary.get('log_data',False),
            fill_low_obs_threshold=dictionary.get('fill_low_obs_threshold'),
            validation_type= dictionary.get("validation_type"),
            frequency_threshold= dictionary.get("frequency_threshold")
        )

    @classmethod
    def create_default_configuration(cls,**kwargs):
        """
        :param kwargs:
        :return:
        """
        return cls(
            owner=kwargs.pop("owner"),
            site=kwargs.pop("site"),
            floor=kwargs.pop("floor"),
            data_path=kwargs.pop("data_path", "/sites"),
            pcaComponentsCount=kwargs.pop("pcaComponentsCount", 0),
            filter_beacons=kwargs.pop("filter_beacons", None),
            observed_value=kwargs.pop("observed_value", -105.0),
            missing_value=kwargs.pop("missing_value", -105.0),
            features=kwargs.pop("features", ["median", "quant3"]),
            beacons=kwargs.pop("beacons", None),
            dropObs=kwargs.pop("dropObs", True),
            channel=kwargs.get("channel", "0"),
            window_size=kwargs.pop("window_size", 5),
            window_step=kwargs.pop("window_step", 5),
            log_data = kwargs.pop("log_data", False),
            fill_low_obs_threshold=kwargs.pop("fill_low_obs_threshold", 30),
            validation_type = kwargs.pop("validation_type",'expand'),
            frequency_threshold = kwargs.pop("frequency_threshold", 2.0)
        )

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not os.path.isdir(value):
            print("path does not exist, creating path:{}".format(value))
            os.mkdir(value)
        self._path = value

    @property
    def beacons(self):
        if self.channel != '0' and self._beacons:
            return [beacon for beacon in self._beacons if re.match('^' + self.channel, beacon)]
        else:
            return self._beacons

    @beacons.setter
    def beacons(self, values):

        self._beacons = [beacon for beacon in values] if values else None

    def get_path(self, suffix=None):
        if suffix is None:
            return self.path
        else:
            return os.path.join(self.path, suffix)

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if value in ['0', '25', '26', '27']:
            self._channel = value
        else:
            raise ValueError("channel is not valid")


    @property
    def validation_type(self):
        return self._validation_type

    @validation_type.setter
    def validation_type(self, value):
        if value in ['roll', 'expand']:
            self._validation_type = value
        else:
            raise ValueError("validation type is not valid")

    @property
    def identifier(self):
        settings = {
            "pcaComponentsCount": int(self.pca_components_count),
            "window_size": np.round(float(self.window_size),2),
            "window_step": np.round(float(self.window_step),2),
            "beacons": str(self.beacons),
            "dropObs": str(self.drop_low_observations),
            "missing_value": np.round(float(self.missing_value),2),
            "observed_value": np.round(float(self.observed_value),2),
            "features": str(self.features),
            "filter_beacons": self.filter_beacons,
            "channel": str(self.channel),
            "validation_type": str(self.validation_type),
            "fill_low_obs_threshold":int(self.fill_low_obs_threshold),
            "frequency_threshold": np.round(float(self.frequency_threshold),2),
            "owner":str(self.owner),
            "site": str(self.site),
            "floor":str(self.floor),
            "log_data": str(self.log_data),
        }
        settings = OrderedDict(sorted(settings.items(), key=lambda t: t[0]))
        identifier_str = "_".join([str(v) + "-" + str(k) for k, v in iteritems(settings)])
        identifier_bytes = identifier_str.encode('utf-8')
        identifier = hashlib.sha1(identifier_bytes).hexdigest()
        return identifier


    def __eq__(self, other):
        if self.identifier == other.identifier:
            return True
        else:
            return False

    def __hash__(self):
        return self.identifier

    @property
    def image_path(self):
        return os.path.join(os.path.split(self.get_path())[0],"image/floor_plan.png")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, values):
        self._features = [feature for feature in values if feature in ['max', 'median','quant3', 'frequency', 'observed']]

    def check_processed_file_exists(self):

        save_directory = self.get_path('processed')
        if self.channel == '0':
            operating_system = 'ios'
        else:
            operating_system = 'android'

        filename = os.path.join(save_directory, '{}-processed-data-{}-{}-{}-{}.pkl'.format(
            'training',
            operating_system,
            self.channel,
            self.pca_components_count,
            self.identifier
        ))
        print("checking for filename: {}".format(filename), end='. ')
        files = glob.glob(filename)
        if files:
            print("file exists")
            return True
        else:
            print ("file doesn't exist")
            return False

    def __init__(self, **kwargs):

        self.data_path = kwargs.pop("data_path")
        self.site = kwargs.pop("site")
        self.owner = kwargs.pop("owner")
        self.floor = kwargs.pop("floor")
        self.path = os.path.join(
            self.data_path,
            self.owner,
            self.site,
            self.floor,
            "fingerprints")

        self.beacons = kwargs.pop("beacons")
        self.missing_value = kwargs.pop("missing_value")
        self.pca_components_count = kwargs.pop("pcaComponentsCount")
        self.drop_low_observations = kwargs.pop("dropObs")
        self.observed_value = kwargs.pop("observed_value")
        self.window_size = kwargs.pop("window_size")
        self.window_step = kwargs.pop("window_step")
        self.features = kwargs.pop("features")
        self.filter_beacons = kwargs.pop("filter_beacons")
        self.log_data = kwargs.pop("log_data")
        self.fill_low_obs_threshold = kwargs.pop("fill_low_obs_threshold")
        self.channel = kwargs.pop("channel", "0")
        self.validation_type = kwargs.pop("validation_type","expand")
        self.frequency_threshold = kwargs.pop("frequency_threshold")

    def __repr__(self):
        return """
        ProcessingConfiguration(
            owner={},
            site={},
            floor={},
            data_path={},
            pcaComponentsCount={},
            filter_beacons={},
            observed_value={},
            missing_value={},
            features={},
            beacons={}, size:{}
            dropObs={},
            frequency_threshold={},
            fill_low_obs_threshold={},
            channel={},
            window_size={},
            window_step={},
            validation_type={},
        )
        identifier = {}
        """.format(self.owner,
                   self.site,
                   self.floor,
                   self.data_path,
                   self.pca_components_count,
                   self.filter_beacons,
                   self.observed_value,
                   self.missing_value,
                   self.features,
                   self.beacons,
                   len(self.beacons),
                   self.drop_low_observations,
                   self.frequency_threshold,
                   self.fill_low_obs_threshold,
                   self.channel,
                   self.window_size,
                   self.window_step,
                   self.validation_type,
                   self.identifier
                   )
