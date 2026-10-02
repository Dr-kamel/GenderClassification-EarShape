import shutil
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from configs.config import (
    CSV_PATH,
    IMAGE_DIR,
    SPLIT_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    RANDOM_STATE
)

def load_metadata():
    """
    Load the IR-EAR metadata CSV file.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Number of records: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


def validate_columns(df):
    """
    Check whether all required columns exist.
    """

    required_columns = [
        "ID",
        "ID-Right",
        "ID-Left",
        "Gender"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_gender_consistency(df):
    """
    Verify that each participant has a single gender label.
    """

    gender_check = (
        df.groupby("ID")["Gender"]
        .nunique()
    )

    invalid_ids = gender_check[
        gender_check > 1
    ].index.tolist()

    if invalid_ids:

        print("\nInconsistent gender labels found:")

        print(
            df[
                df["ID"].isin(invalid_ids)
            ][
                ["ID", "Gender"]
            ]
            .sort_values("ID")
        )

        raise ValueError(
            f"Found {len(invalid_ids)} participants "
            f"with inconsistent gender labels."
        )

    print("\nGender consistency check: PASSED")


def prepare_labels(df):
    """
    Clean and convert gender labels.

    Men   -> 0
    Women -> 1
    """

    df = df[
        [
            "ID",
            "ID-Right",
            "ID-Left",
            "Gender"
        ]
    ].copy()

    df = df.dropna(
        subset=[
            "ID",
            "Gender"
        ]
    )

    gender_values = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    gender_mapping = {
        "man": 0,
        "woman": 1
    }

    unknown_values = (
        set(gender_values)
        - set(gender_mapping)
    )

    if unknown_values:
        raise ValueError(
            f"Unknown gender values: "
            f"{sorted(unknown_values)}"
        )

    df["Gender"] = (
        gender_values
        .map(gender_mapping)
        .astype(int)
    )

    if not df["Gender"].isin([0, 1]).all():
        raise ValueError(
            "Gender column must contain only 0 and 1."
        )

    return df

def create_person_table(df):
    """
    Create one record per participant.
    """

    persons = (
        df[
            ["ID", "Gender"]
        ]
        .drop_duplicates(
            subset="ID"
        )
        .copy()
    )

    if len(persons) != df["ID"].nunique():
        raise ValueError(
            "Participant-level table contains duplicates."
        )

    return persons


def split_participants(persons):
    """
    Perform participant-level stratified splitting.

    80% training
    10% validation
    10% testing
    """

    person_ids = persons["ID"].values

    person_gender = persons["Gender"].values

    train_ids, temp_ids = train_test_split(

        person_ids,

        test_size=(
            VAL_SIZE + TEST_SIZE
        ),

        random_state=RANDOM_STATE,

        stratify=person_gender
    )

    temp_persons = persons[
        persons["ID"].isin(temp_ids)
    ].copy()

    val_ids, test_ids = train_test_split(

        temp_persons["ID"].values,

        test_size=(
            TEST_SIZE
            /
            (VAL_SIZE + TEST_SIZE)
        ),

        random_state=RANDOM_STATE,

        stratify=temp_persons["Gender"]
    )

    return (
        train_ids,
        val_ids,
        test_ids
    )

def print_split_information(
    persons,
    train_ids,
    val_ids,
    test_ids
):

    train_persons = persons[
        persons["ID"].isin(train_ids)
    ]

    val_persons = persons[
        persons["ID"].isin(val_ids)
    ]

    test_persons = persons[
        persons["ID"].isin(test_ids)
    ]

    print("\n" + "=" * 60)
    print("PARTICIPANT-LEVEL SPLIT")
    print("=" * 60)

    print(
        f"Total participants: "
        f"{len(persons)}"
    )

    print(
        f"Training participants: "
        f"{len(train_persons)}"
    )

    print(
        f"Validation participants: "
        f"{len(val_persons)}"
    )

    print(
        f"Test participants: "
        f"{len(test_persons)}"
    )

    print("\nTraining gender distribution:")
    print(
        train_persons["Gender"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation gender distribution:")
    print(
        val_persons["Gender"]
        .value_counts()
        .sort_index()
    )

    print("\nTest gender distribution:")
    print(
        test_persons["Gender"]
        .value_counts()
        .sort_index()
    )

def create_split_directories():

    for split in [
        "train",
        "validation",
        "test"
    ]:

        for gender in [
            "0",
            "1"
        ]:

            directory = (
                SPLIT_DIR
                / split
                / gender
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


def copy_ear_image(
    image_id,
    gender,
    split
):
    """
    Copy an ear image to the corresponding split folder.
    """

    if pd.isna(image_id):
        return

    image_id = int(image_id)

    filename = f"{image_id}.png"

    source = IMAGE_DIR / filename

    destination = (
        SPLIT_DIR
        / split
        / str(gender)
        / filename
    )

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

    else:

        print(
            f"Warning: file not found: {source}"
        )

def copy_dataset(
    df,
    train_ids,
    val_ids,
    test_ids
):

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    for _, row in df.iterrows():

        person_id = row["ID"]

        gender = int(row["Gender"])

        if person_id in train_ids:

            split = "train"

        elif person_id in val_ids:

            split = "validation"

        elif person_id in test_ids:

            split = "test"

        else:

            continue

        copy_ear_image(
            row["ID-Right"],
            gender,
            split
        )

        copy_ear_image(
            row["ID-Left"],
            gender,
            split
        )


def print_image_counts():

    print("\n" + "=" * 60)
    print("IMAGE COUNTS")
    print("=" * 60)

    for split in [
        "train",
        "validation",
        "test"
    ]:

        for gender in [
            "0",
            "1"
        ]:

            folder = (
                SPLIT_DIR
                / split
                / gender
            )

            if not folder.exists():
                count = 0

            else:

                count = len([
                    file
                    for file in folder.iterdir()
                    if file.suffix.lower()
                    in [
                        ".png",
                        ".jpg",
                        ".jpeg"
                    ]
                ])

            print(
                f"{split:12s} "
                f"Gender {gender}: "
                f"{count}"
            )



def main():

    df = load_metadata()

    validate_columns(df)

    validate_gender_consistency(df)

    df = prepare_labels(df)

    print("\nGender distribution:")
    print(
        df["Gender"]
        .value_counts()
        .sort_index()
    )

    persons = create_person_table(df)

    print(
        f"\nTotal participants: "
        f"{len(persons)}"
    )

    print(
        f"Men: "
        f"{(persons['Gender'] == 0).sum()}"
    )

    print(
        f"Women: "
        f"{(persons['Gender'] == 1).sum()}"
    )

    (
        train_ids,
        val_ids,
        test_ids
    ) = split_participants(persons)

    print_split_information(
        persons,
        train_ids,
        val_ids,
        test_ids
    )

    create_split_directories()

    copy_dataset(
        df,
        train_ids,
        val_ids,
        test_ids
    )

    print_image_counts()

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()