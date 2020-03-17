import forest.drivers


def test_dataset():
    settings = {
        "pattern": "*.nc"
    }
    dataset = forest.drivers.get_dataset("unified_model", settings)
