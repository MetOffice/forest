import datetime as dt
import forest.components.time


def test_time_ui_render():
    time = dt.datetime(2020, 1, 1)
    ui = forest.components.time.TimeUI()
    ui.render(time, [time])
