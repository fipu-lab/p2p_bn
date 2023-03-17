import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import environ
environ.check_devices()
if environ.is_collab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from models.util import *
from models.abstract_model import *

