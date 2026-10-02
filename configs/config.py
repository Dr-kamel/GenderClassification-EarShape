from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

CSV_PATH = DATA_DIR / "IR-EAR dataset.csv"

SPLIT_DIR = DATA_DIR / "split"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
HISTORY_DIR = RESULTS_DIR / "histories"

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

RANDOM_STATE = 42

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

EPOCHS = 50

DROPOUT_RATE = 0.5

DENSE_UNITS = 512

EARLY_STOPPING_PATIENCE = 10

REDUCE_LR_PATIENCE = 5

REDUCE_LR_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-7

CLASS_NAMES = {
    0: "Men",
    1: "Women"
}

for directory in [
    SPLIT_DIR,
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    HISTORY_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    