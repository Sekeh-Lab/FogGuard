import pathlib
import datetime

# cur_time = datetime.datetime.now().strftime("%d-%m-%H-%M-%S")
cur_time = datetime.datetime.now().strftime("%H-%M")
# cur_time = ""
ROOT = str(pathlib.Path(__file__).resolve().parents[1]) + "/"
DATA_DIR = ROOT + "data/"
SRC_DIR = ROOT + "src/"
OUTPUT_DIR = ROOT + "output/"
CHPT_DIR = ROOT + "checkpoints/"
RUN_DIR = OUTPUT_DIR # + cur_time + "/" 
# MODEL_ROOT_DIR = OUTPUT_DIR + "model/"
MODEL_ROOT_DIR = RUN_DIR # + "model/"
LOG_CONFIG_DIR = SRC_DIR + "logging.ini"
