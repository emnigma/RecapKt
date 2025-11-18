from enum import Enum


class MetricType(Enum):
    COHERENCE = "COHERENCE"
    F1_TOOL = "F1_TOOL"


class ModelPriceForMillionTokensInDollars(Enum):
    GPT_3_5_TURBO_INPUT = 0.50
    GPT_3_5_TURBO_OUTPUT = 1.50
    GPT_4_1_MINI_INPUT = 0.40
    GPT_4_1_MINI_OUTPUT = 1.60
    GPT_4_O_INPUT = 2.50
    GPT_4_O_OUTPUT = 10.00
    GPT_4_1_INPUT = 2.00
    GPT_4_1_OUTPUT = 8.00
    GPT_5_NANO_INPUT = 0.05
    GPT_5_NANO_OUTPUT = 0.40
    GPT_5_MINI_INPUT = 0.250
    GPT_5_MINI_OUTPUT = 2.000

