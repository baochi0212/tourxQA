from transformers import (
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
)



MODEL_MAP = {
    'IDSF': [IDSFModule],
    'QA': [QAModule]
}


CONFIG_MAP = {
    "xlmr": (XLMRobertaConfig, XLMRobertaTokenizer),
    "phobert": (RobertaConfig, AutoTokenizer),
}

PRETRAINED_MAP = {
    "xlmr": "xlm-roberta-base",
    "phobert": "vinai/phobert-base",
}