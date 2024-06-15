"""
Getting model by alias.
"""

from .models import (
    CommonMachineLearning,
    Voting,
    JeongStacking,
    ARIMA,
    RecurrentNetwork,
    LongShortTermMemory,
    CNN1D,
    CNNLSTM,
    GANs,
    TransformerTS,
    KAN
)


def get_by_alias(alias: str, **kwargs):
    """
    Get model by alias. This function will take alias and return the model.
    """
    def filter_and_format(alias: str, kws: dict):
        # Filter out the keys that are not in the model's parameters
        return {k.split('_', 1)[1]: v for k, v in kws.items() if alias in k}

    if alias == 'lr':
        return CommonMachineLearning(**filter_and_format('lr', kwargs))
    if alias == 'knn':
        return CommonMachineLearning(model_alias="knn", **filter_and_format('knn', kwargs))
    if alias == 'svm':
        return CommonMachineLearning(model_alias="svm", **filter_and_format('svm', kwargs))
    if alias == 'dt':
        return CommonMachineLearning(model_alias="dt", **filter_and_format('dt', kwargs))
    if alias == 'et':
        return CommonMachineLearning(model_alias="et", **filter_and_format('et', kwargs))
    if alias == 'ada':
        return CommonMachineLearning(model_alias="ada", **filter_and_format('ada', kwargs))
    if alias == 'bag':
        return CommonMachineLearning(model_alias="bag", **filter_and_format('bag', kwargs))
    if alias == 'gb':
        return CommonMachineLearning(model_alias="gb", **filter_and_format('gb', kwargs))
    if alias == 'rf':
        return CommonMachineLearning(model_alias="rf", **filter_and_format('rf', kwargs))
    if alias == 'xgb':
        return CommonMachineLearning(model_alias="xgb", **filter_and_format('xgb', kwargs))
    if alias == 'vote':
        return Voting(**filter_and_format('vote', kwargs))
    if alias == 'jeong':
        return JeongStacking(**filter_and_format('jeong', kwargs))
    if alias == 'arima':
        return ARIMA(**filter_and_format('arima', kwargs))
    if alias == 'rnn':
        return RecurrentNetwork(**filter_and_format('rnn', kwargs))
    if alias == 'lstm':
        return LongShortTermMemory(**filter_and_format('lstm', kwargs))
    if alias == 'cnn1d':
        return CNN1D(**filter_and_format('cnn1d', kwargs))
    if alias == 'cnnlstm':
        return CNNLSTM(**filter_and_format('cnnlstm', kwargs))
    if alias == 'gans':
        return GANs(**filter_and_format('gans', kwargs))
    if alias == 'transformer':
        return TransformerTS(**filter_and_format('transformer', kwargs))
    if alias == 'kan':
        return KAN(**filter_and_format('kan', kwargs))
    raise ValueError(
        f'Invalid model alias. Got: {alias}')


def get_by_aliases(aliases: list[str], **kwargs):
    """
    Get models by aliases.
    """
    return [get_by_alias(alias, **kwargs) for alias in aliases]
