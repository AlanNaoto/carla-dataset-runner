"""
Defines a few custom weather presets. Up until version 0.9.6, carla has a few native ones; however, on my data capture
setup they don't seem to differ that much one from another
"""
import sys
import os
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'CARLA_0.9.6')
CARLA_EGG_PATH = os.path.join(CARLA_DIR, 'PythonAPI', 'carla', 'dist', 'carla-0.9.6-py3.5-linux-x86_64.egg')
sys.path.append(CARLA_EGG_PATH)
import carla

# https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters-class
class WeatherSelector:
    def __init__(self):
        self.cloudiness = None  # 0.0 to 100.0
        self.precipitation = None  # 0.0 to 100.0
        self.precipitation_deposits = None  # 0.0 to 100.0
        self.wind_intensity = None  # 0.0 to 100.0
        self.sun_azimuth_angle = None  # 0.0 to 360.0
        self.sun_altitude_angle = None  # -90.0 to 90.0

    def get_weather_options(self):
        return [self.morning(), self.midday(), self.afternoon(), self.sunny_rain(), self.almost_night()]

    def morning(self):
        self.cloudiness = 20.0
        self.precipitation = 90.0
        self.precipitation_deposits = 30.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = 30.0
        return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
                self.sun_azimuth_angle, self.sun_altitude_angle]

    def midday(self):
        self.cloudiness = 30.0
        self.precipitation = 0.0
        self.precipitation_deposits = 60.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 00.0
        self.sun_altitude_angle = 45.0
        return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
                self.sun_azimuth_angle, self.sun_altitude_angle]

    def afternoon(self):
        self.cloudiness = 50.0
        self.precipitation = 0.0
        self.precipitation_deposits = 40.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = -40.0
        return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
                self.sun_azimuth_angle, self.sun_altitude_angle]

    def sunny_rain(self):
        self.cloudiness = -1.0
        self.precipitation = -1.0
        self.precipitation_deposits = -1.0
        self.wind_intensity = -1.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = -1.0
        return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
                self.sun_azimuth_angle, self.sun_altitude_angle]

    def almost_night(self):
        self.cloudiness = 30.0
        self.precipitation = 30.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = -60.0
        return [self.cloudiness, self.precipitation, self.precipitation_deposits, self.wind_intensity,
                self.sun_azimuth_angle, self.sun_altitude_angle]
