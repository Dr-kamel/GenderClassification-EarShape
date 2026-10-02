import tensorflow as tf
import pandas as pd


from configs.config import (
    SPLIT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_STATE
)

from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess
)


from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess
)

from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocess
)

from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inception_resnet_v2_preprocess
)

PREPROCESS_FUNCTIONS = {

    "VGG16":
        vgg16_preprocess,

    "DenseNet121":
        densenet_preprocess,

    "EfficientNetB7":
        None,

    "EfficientNetB0":
        None,

    "Xception":
        xception_preprocess,

    "InceptionResNetV2":
        inception_resnet_v2_preprocess,

    "ConvNeXt-Tiny":
        tf.keras.applications.convnext.preprocess_input,

    "MobileNetV3-Large":
        tf.keras.applications.mobilenet_v3.preprocess_input
}

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomRotation(
            factor=0.08
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15)
        ),

        tf.keras.layers.RandomBrightness(
            factor=0.20
        )

    ],

    name="data_augmentation"
)

def build_dataframe(split_name):
    """
    Create a dataframe containing image paths and labels.
    """

    image_paths = []
    labels = []

    split_path = SPLIT_DIR / split_name

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_path}"
        )

    for gender in ["0", "1"]:

        class_path = (
            split_path / gender
        )

        if not class_path.exists():
            continue

        for file in sorted(
            class_path.iterdir()
        ):

            if file.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg"
            ]:

                image_paths.append(
                    str(file)
                )

                labels.append(
                    int(gender)
                )

    dataframe = pd.DataFrame(
        {
            "path": image_paths,
            "gender": labels
        }
    )

    return dataframe


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label

def get_preprocess_function(model_name):

    if model_name not in PREPROCESS_FUNCTIONS:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return PREPROCESS_FUNCTIONS[
        model_name
    ]

def create_dataset(
    dataframe,
    model_name,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            dataframe["path"].values,
            dataframe["gender"].values
        )

    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.shuffle(

            buffer_size=len(dataframe),

            seed=RANDOM_STATE,

            reshuffle_each_iteration=True
        )

        dataset = dataset.map(

            lambda image, label: (

                data_augmentation(
                    image,
                    training=True
                ),

                label
            ),

            num_parallel_calls=tf.data.AUTOTUNE
        )

    preprocess_fn = get_preprocess_function(
        model_name
    )

    if preprocess_fn is not None:

        dataset = dataset.map(

            lambda image, label: (

                preprocess_fn(image),

                label
            ),

            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def load_dataframes():

    train_df = build_dataframe(
        "train"
    )

    val_df = build_dataframe(
        "validation"
    )

    test_df = build_dataframe(
        "test"
    )

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    val_df = val_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    return (
        train_df,
        val_df,
        test_df
    )

def print_dataset_information(
    train_df,
    val_df,
    test_df
):

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(
        f"Training images: "
        f"{len(train_df)}"
    )

    print(
        f"Validation images: "
        f"{len(val_df)}"
    )

    print(
        f"Test images: "
        f"{len(test_df)}"
    )

    print("\nTraining distribution:")
    print(
        train_df["gender"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation distribution:")
    print(
        val_df["gender"]
        .value_counts()
        .sort_index()
    )

    print("\nTest distribution:")
    print(
        test_df["gender"]
        .value_counts()
        .sort_index()
    )