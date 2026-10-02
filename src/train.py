import argparse

import pandas as pd
import tensorflow as tf


from configs.config import (
    EPOCHS,
    HISTORY_DIR
)

from dataset import (
    load_dataframes,
    create_dataset,
    print_dataset_information
)

from models import (
    AVAILABLE_MODELS,
    build_model,
    create_callbacks
)

def train_model(model_name):

    print("\n" + "=" * 70)
    print(
        f"TRAINING MODEL: {model_name}"
    )
    print("=" * 70)

    (
        train_df,
        val_df,
        test_df
    ) = load_dataframes()

    print_dataset_information(
        train_df,
        val_df,
        test_df
    )

    train_dataset = create_dataset(

        train_df,

        model_name=model_name,

        training=True
    )

    validation_dataset = create_dataset(

        val_df,

        model_name=model_name,

        training=False
    )

    model = build_model(
        model_name
    )

    model.summary()

    callbacks = create_callbacks(
        model_name
    )

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    history_df = pd.DataFrame(
        history.history
    )

    history_path = (
        HISTORY_DIR
        / f"{model_name}_history.csv"
    )
    

    history_df.to_csv(
        history_path,
        index=False
    )

    print(
        f"\nTraining history saved to:"
        f"\n{history_path}"
    )

    print(
        f"\nTraining completed for "
        f"{model_name}"
    )

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Train an ImageNet-pretrained "
            "CNN for ear-based gender classification."
        )
    )

    parser.add_argument(

        "--model",

        type=str,

        required=True,

        choices=AVAILABLE_MODELS,

        help="Model architecture to train."
    )

    args = parser.parse_args()

    train_model(
        args.model
    )


if __name__ == "__main__":
    main()