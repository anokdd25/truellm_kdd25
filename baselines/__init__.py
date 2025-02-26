# __init__.py

from baseline_general import run_baselines

__all__ = ['run_baselines']
DEFAULT_CONFIG = {'models_available': {'classic': ('dnn', 'log', 'dt', 'rf', 'xgboost'), 'aix': ('glrm', )}}