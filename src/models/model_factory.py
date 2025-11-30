"""Factory for creating model instances."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

# XGBoost will be imported lazily only when needed
XGBOOST_AVAILABLE = None

def _check_xgboost_available():
    """Check if XGBoost is available (lazy import)."""
    global XGBOOST_AVAILABLE
    if XGBOOST_AVAILABLE is None:
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
            logger.warning("XGBoost not available. Install with: pip install xgboost")
        except Exception as e:
            XGBOOST_AVAILABLE = False
            logger.warning(f"XGBoost not available due to: {str(e)}. "
                          "On macOS, you may need to install OpenMP: brew install libomp")
    return XGBOOST_AVAILABLE


def create_model(model_type: str, hyperparameters: Dict[str, Any], random_state: int = 42):
    """
    Create a model instance based on type and hyperparameters.
    
    Args:
        model_type: Type of model ('logistic_regression', 'svm', 'random_forest', 'xgboost')
        hyperparameters: Dictionary of hyperparameters
        random_state: Random seed
        
    Returns:
        Model instance
    """
    hyperparameters = hyperparameters.copy()
    hyperparameters['random_state'] = random_state
    
    if model_type == "logistic_regression":
        return LogisticRegression(**hyperparameters)
    
    elif model_type == "svm":
        return SVC(**hyperparameters, probability=True)
    
    elif model_type == "random_forest":
        return RandomForestClassifier(**hyperparameters)
    
    elif model_type == "xgboost":
        if not _check_xgboost_available():
            raise ImportError(
                "XGBoost is not available. Install with: pip install xgboost\n"
                "On macOS, you may also need: brew install libomp"
            )
        import xgboost as xgb
        return xgb.XGBClassifier(**hyperparameters)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

