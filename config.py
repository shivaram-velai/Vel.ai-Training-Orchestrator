import os

class DirectoryConfig:
    def __init__(self):
        self.raw = RawDirectoryConfig()
        self.features = FeaturesDirectoryConfig()

class RawDirectoryConfig:
    def __init__(self):
        self.price_dataset = ""
        self.universe_definition_file = ""
        self.market_dataset = ""
        self.sector_mapping_file = ""

class FeaturesDirectoryConfig:
    def __init__(self):
        self.events_open_dataset = ""
        self.events_close_dataset = ""
        self.events_master_dataset = ""

class ExperimentConfig:
    def __init__(self):
        self.experiment_name = 'Live_System_2026'
        self.prediction_horizon = 7
        self.trading_calendar = 'NYSE'
        self.test_period_months = 1
        self.validation_period_months = 1
        self.validation_offset_months = 0
        self.test_period_start_offset_months = 0

class PipelineSettingsConfig:
    def __init__(self):
        self.apply_clip_on_return = False
        self.apply_market_neutral = False
        self.beta_based_market_neutral = False
        self.beta_based_market_neutral_config = BetaBasedMarketNeutralConfig()

class BetaBasedMarketNeutralConfig:
    def __init__(self):
        self.full_window = 0
        self.min_window = 0
        self.shift_window = 0
        self.beta_cap_min = 0.0
        self.beta_cap_max = 0.0
        self.shrink_toward = 0.0
        self.shrink_weight = 0.0

class TransformationsConfig:
    def __init__(self):
        self.lower_bound = 0.0
        self.upper_bound = 0.0

class Config:
    def __init__(self):
        self.directory = DirectoryConfig()
        self.experiment = ExperimentConfig()
        self.pipeline_settings = PipelineSettingsConfig()
        self.transformations = TransformationsConfig()
        self.prediction_s3_bucket = 'prediction-outputs-s3'