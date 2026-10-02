import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.applications import (
    VGG16,
    DenseNet121,
    EfficientNetB7,
    EfficientNetB0,
    Xception,
    InceptionResNetV2,
    ConvNeXtTiny,
    MobileNetV3Large
)

from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


from configs.config import (
    IMG_SIZE,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    MODEL_DIR
)

AVAILABLE_MODELS = [

    "VGG16",

    "DenseNet121",

    "EfficientNetB7",

    "EfficientNetB0",

    "Xception",

    "InceptionResNetV2",

    "ConvNeXt-Tiny",

    "MobileNetV3-Large"
]

def get_backbone(model_name):

    input_shape = (
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )

    if model_name == "VGG16":

        backbone = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "DenseNet121":

        backbone = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "EfficientNetB7":

        backbone = EfficientNetB7(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "EfficientNetB0":

        backbone = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "Xception":

        backbone = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "InceptionResNetV2":

        backbone = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "ConvNeXt-Tiny":

        backbone = ConvNeXtTiny(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

    elif model_name == "MobileNetV3-Large":

        backbone = MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return backbone


def build_model(model_name):

    backbone = get_backbone(
        model_name
    )

    backbone.trainable = False

    inputs = Input(
        shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        )
    )

    x = backbone(
        inputs,
        training=False
    )

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        DENSE_UNITS,
        activation="relu"
    )(x)

    x = Dropout(
        DROPOUT_RATE
    )(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model

def create_callbacks(model_name):

    model_path = (
        MODEL_DIR
        / f"best_{model_name}.keras"
    )

    checkpoint = ModelCheckpoint(

        filepath=str(model_path),

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    )

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=True,

        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(

        monitor="val_loss",

        factor=REDUCE_LR_FACTOR,

        patience=REDUCE_LR_PATIENCE,

        min_lr=MIN_LEARNING_RATE,

        verbose=1
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr
    ]