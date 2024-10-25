import sys
sys.path.append(".")

from .config_mamba import MambaConfig
# from .mixer_seq_simple import MambaLMHeadModel
from .model import MambaLMHeadModel, scMambaLMHeadModel