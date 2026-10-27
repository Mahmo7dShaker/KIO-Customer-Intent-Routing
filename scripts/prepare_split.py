from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz #علشان احسب لبتشبه بين الtxt
import random

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "kio_60_messages_website_development_only.csv"
df = pd.read_csv(data_path, encoding="utf-8")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "-" * 50)

print("\nDataset info:")
df.info()

print("\n" + "-" * 50)

# show class distribution
print("\nClass Distribution:")
print(df["label"].value_counts().sort_index())

print("\n" + "-" * 50)

# show case type distribution
print("\nCase Types:")
print(df["case_type"].value_counts())

print("\n" + "-" * 50)

# show case types by label
print("\nCase types by label:")
print(pd.crosstab(df["case_type"], df["label"]))

print("\n" + "-" * 50)

# show missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\n" + "-" * 50)

normalized_messages = (
    df["message"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\nDuplicate messages:")
print(normalized_messages.duplicated().sum())

print("\n" + "-" * 50)

# show group statistics
print("\nTotal Groups:")
print(df["group_id"].nunique())

print("\nGroup Sizes:")
print(df.groupby("group_id").size().value_counts().sort_index())

print("\nGroup with Conflicting Labels:")
print((df.groupby("group_id")["label"].nunique() > 1).sum())

print("\nParaphrase Groups:")
for group_id, group in df.groupby("group_id"):
    if len(group) > 1:
        print(f"\nGroup: {group_id}")
        print(group[["id", "message", "label"]].to_string(index=False))

print("\n" + "-" * 50)


print("\nSemantic Group Audit:")

pairs_to_check = [
    ("M016", "M023"),
    ("M016", "M051"),
    ("M025", "M056"),
    ("M001", "M013"),
    ("M018", "M046"),
    ("M030", "M031"),
]

df_by_id = df.set_index("id")

for id1, id2 in pairs_to_check:
    row1 = df_by_id.loc[id1]
    row2 = df_by_id.loc[id2]

    score = fuzz.ratio(
        row1["message"].lower(),
        row2["message"].lower()
    )

    print(f"\n{id1} vs {id2}")
    print(f"Similarity: {score:.1f}%")
    print("Group IDs:", row1["group_id"], row2["group_id"])
    print("Message 1:", row1["message"])
    print("Message 2:", row2["message"])

print("\n" + "-" * 50)

group_ubdate = {
    "M051": "SP01",
    "M056": "S05",
    "M001": "S01",
    "M046": "S10",
    "M031": "S07",
}
for messge_id, new_group in group_ubdate.items():
    df.loc[df["id"] == messge_id, "group_id"] = new_group

print("\nUpdated Group Analysis:")
print("\nTotal Groups:", df["group_id"].nunique())
print("\nGroup Sizes:")
print(df.groupby("group_id").size().value_counts().sort_index())

print("\n" + "-" * 50)

print("\nConflicting Labels:")
print((df.groupby("group_id")["label"].nunique() > 1).sum())

print("\n" + "-" * 50)

print("\nCass Distribution:")
print(df["label"].value_counts().sort_index())

print("\n" + "-" * 50)


#random generator
rng = random.Random(351)

test_group = set()

# Process each class separately to keep the split balanced
for label in (0, 1):

    class_rows = df[df["label"] == label]

    # Group related messages and convert the groups to a list
    groups = list(class_rows.groupby("group_id"))

    # Get all case type
    class_case_types = set(class_rows["case_type"])

    found_split = False

    for attempt in range(2000):

        # Shuffle whole groups
        rng.shuffle(groups)

        selected = []
        count = 0

        # Examine one complete group at a time
        for group_id, group in groups:

            # Add the whole group only if total stays <= 10
            if count + len(group) <= 10:
                selected.append(group_id)
                count += len(group)
            if count == 10:
                break

        # Get test messages from the selected groups
        test_rows = class_rows[
            class_rows["group_id"].isin(selected)
        ]

        # Get training messages
        train_row = class_rows[
            ~class_rows["group_id"].isin(selected)
        ]

        # Accept only an exact, case-diverse split
        if (
            len(test_rows) == 10
            and set(test_rows["case_type"]) == class_case_types
            and set(train_row["case_type"]) == class_case_types
        ):

            # Save valid test group IDs
            test_group.update(selected)

            found_split = True
            break

    if not found_split:
        raise ValueError(f"No valid split for label {label}")


# Assign every message to train or test by group ID
df["split"] = df["group_id"].map(
    lambda group_id: "test" if group_id in test_group else "train"
)


# Simplify Message IDs 
# M001 -> M01, M010 -> M10
df["id"] = df["id"].str.replace(
    r"^M0(?=\d{2}$)", "M", regex=True
)

# Create sequential Group IDs / إعادة تسمية
# SP01, S05, TP03 -> G01, G02, ...
old_groups = sorted(df["group_id"].unique())

# Map old groups to new IDs 
group_mapping = {
    old: f"G{i:02d}"
    for i, old in enumerate(old_groups, start=1)
}

# Apply new Group IDs
df["group_id"] = df["group_id"].map(group_mapping)

# Show updated IDs
print("\nSimplified IDs:")
print(df[["id", "group_id", "split"]].head(10))


# Show class counts for train and test
print("\nTrain/Test Distribution:")
print(pd.crosstab(df["split"], df["label"]))

# Show case-type
print("\nCase Type by Split:")
print(pd.crosstab(
    [df["split"], df["case_type"]],
    df["label"]
))


print("\n" + "-" * 50)
print("\nFinal Dataset Validation:")

assert len(df) == 60, "Datadet must have 60 messages"

#check valid labels
assert df["id"].is_unique, "Duplicate message IDs found"

#check empty messages
assert df["message"].notna().all()
assert df["message"].str.strip().ne("").all()

#check exact split sizes
assert (df["split"] == "train").sum() == 40, "Train set must have 40 messages"
assert (df["split"] == "test").sum() == 20, "Test set must have 20 messages"

# check class balance
counts = pd.crosstab(df["split"], df["label"])

assert counts.loc["train", 0] == 20
assert counts.loc["train", 1] == 20
assert counts.loc["test", 0] == 10
assert counts.loc["test", 1] == 10

assert (df.loc[df["case_type"] == "mixed", "label"] == 0).all()

print("\nBasic validation: PASSED")

#separate train and test
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

#collect group IDs
train_groups = set(train_df["group_id"])
test_groups = set(test_df["group_id"])

#find overlapping groups
overlab = train_groups.intersection(test_groups)

print("\nOverlapping Groups:", overlab)

#stop if any overlap found
assert len(overlab) == 0, "Group leakage detected!"

# verify each group has only one split
assert df.groupby("group_id")["split"].nunique().eq(1).all()

print("\nGroup leakage check: PASSED")

similarity_pairs = []

# Compare train messages against test
for _, train_row in train_df.iterrows():
    for _, test_row in test_df.iterrows():

        # Normalize text
        text1 = " ".join(train_row["message"].casefold().split())
        text2 = " ".join(test_row["message"].casefold().split())

        # Calculate lexical similarity حساب التشابه 
        score = fuzz.ratio(text1, text2)

        # Save similarity and message 
        similarity_pairs.append((
            score,
            train_row["id"],
            test_row["id"]
        ))

similarity_pairs.sort(reverse=True)

print("\nTop 10 Cross-Split Similarities:")

for score, train_id, test_id in similarity_pairs[:10]:
    print(f"{train_id} vs {test_id}: {score:.1f}%")


# Keep dataset columns
columns = [
    "id", "message", "label", "intent_type",
    "case_type", "group_id", "split"
]

# Save the dataset
output_path = project_dir / "data" / "dataset.csv"

df[columns].to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nFinal dataset saved:", output_path)