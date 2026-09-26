"""Model variants and factory. All wrap or subclass the original models in moving_mnist/."""
from .factory import MODEL_NAMES, TRAINABLE, NEEDS_MOTION, build_model, run_model  # noqa: F401
from .felstm_plus import FEConvLSTMPlus  # noqa: F401
from .gates import bias_for_tau, set_forget_bias  # noqa: F401
from .melstm_plus import MEConvLSTMPlus, MeanFlowConnection  # noqa: F401
from .persistence import Persistence  # noqa: F401
from .stabilizer import OracleStabilized  # noqa: F401
