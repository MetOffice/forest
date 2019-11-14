import forest


def test_driver_spec():
    driver_name = "intake"
    spec = forest.config.DriverSpec(driver_name)
    assert spec.name == driver_name
    assert spec.settings == {}
