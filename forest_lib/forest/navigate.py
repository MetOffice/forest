import numpy as np


__all__ = [
    "TimeControl"
]



class TimeControl(object):
    """Forecast/time navigation controller

    Navigates through forecast/time space, by
    either keeping time fixed, forecast length fixed
    or model run fixed
    """
    def __init__(self, forecast_times):
        self.forecast_times = np.asarray(forecast_times, dtype=object)
        self.run_index = 0
        self.forecast_index = 0

    @property
    def forecast_length(self):
        return self.valid_time - self.model_start_time

    @property
    def model_start_time(self):
        return self.forecast_times[self.run_index, 0]

    @property
    def valid_time(self):
        return self.forecast_times[self.run_index, self.forecast_index]

    def next_forecast(self):
        self.forecast_index += 1

    def previous_forecast(self):
        self.forecast_index -= 1

    def next_run(self):
        if self.run_index == len(self.forecast_times) - 1:
            raise IndexError("already at final run")
        self.run_index += 1

    def next_lead_time(self):
        if self.run_index == 0:
            raise IndexError("already at earliest run")
        valid_time = self.valid_time
        forecasts = self.forecast_times[self.run_index - 1].tolist()
        self.forecast_index = forecasts.index(valid_time)
        self.run_index -= 1

    def most_recent_run(self):
        self.run_index = len(self.forecast_times) - 1
