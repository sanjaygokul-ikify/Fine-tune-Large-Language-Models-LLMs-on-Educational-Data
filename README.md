# Fine-tune-Large-Language-Models-LLMs-on-Educational-Data
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df  = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train_df = train_df.dropna(subset=["StudentExplanation", "Misconception"])
train_df["Misconception"] = train_df["Misconception"].apply(lambda x: x.split())

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(train_df["Misconception"])
train_df["labels"] = list(labels)

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["StudentExplanation"], truncation=True)

hf_train = Dataset.from_pandas(train_df[["StudentExplanation", "labels"]])
hf_train = hf_train.map(tokenize_function, batched=True)
hf_train = hf_train.map(lambda x: {"labels": [float(i) for i in x["labels"]]}, batched=False)
hf_train = hf_train.train_test_split(test_size=0.1)
train_ds = hf_train["train"]
eval_ds  = hf_train["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./misconception_model")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)

tokenizer = AutoTokenizer.from_pretrained("./misconception_model")
model = AutoModelForSequenceClassification.from_pretrained("./misconception_model")
model.eval()

with open("label_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(lambda x: tokenizer(x["StudentExplanation"], truncation=True), batched=True)
test_loader = DataLoader(
    test_ds.remove_columns(["StudentExplanation"]),
    batch_size=16,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
)

predictions = []
for batch in test_loader:
    with torch.no_grad():
        logits = model(**batch).logits
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        for row in preds:
            labels = mlb.inverse_transform([row])[0]
            predictions.append(" ".join(labels))

submission = pd.DataFrame({
    "id": test_df["id"],
    "Misconception": predictions
})
submission.to_csv("submission.csv", index=False)
