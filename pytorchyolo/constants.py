import pathlib
import datetime

# cur_time = datetime.datetime.now().strftime("%d-%m-%H-%M-%S")
cur_time = datetime.datetime.now().strftime("%H-%M")
# cur_time = ""
ROOT = str(pathlib.Path(__file__).resolve().parents[1]) + "/"
DATA_DIR = ROOT + "data/"
OUTPUT_DIR = ROOT + "output/"
RUN_DIR = OUTPUT_DIR # + cur_time + "/" 
# MODEL_ROOT_DIR = OUTPUT_DIR + "model/"
MODEL_ROOT_DIR = RUN_DIR # + "model/"
LOG_CONFIG_DIR = ROOT + "pytorchyolo/logging.ini"
CHPT_DIR = OUTPUT_DIR + "checkpoints/"
