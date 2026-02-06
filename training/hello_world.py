from itertools import batched
#import tensorflow_datasets as tfds
#import torch
#from torch.utils.data import dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
#import tensorflow
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from evaluate import load
import os

#set huggingface token
HF_TOKEN=os.environ["HF_TOKEN"]

#data_files = {"train": "s4s.file"}
#fourchannel = load_dataset("csv", data_files=data_files)
#fourchannel = fourchannel["train"].train_test_split(test_size=0.2, seed=42)
i,j=0,0
fourchannel = []
with open("s4s.file", "r+") as file:
    for line in file:
        j += 1
        if len(line.split(" ")) < 50:
            continue
        fourchannel.append({
            "text": line.strip(),
            "label": 0
        })
        i += 1
        #if i >= 100000:
        #    break
print("\n ====== \n" + str(i)+'/'+str(j) + "("+ str(i/j) +")" + " of 4chan posts used\n ====== \n")
fourchannel = Dataset.from_list(fourchannel)


#LOADING REDDIT DATA!
#redditclean, cols = content, summary 

def datasetloader(name, col_to_keep, label):
    variable = load_dataset(name)
    variable = concatenate_datasets([variable[split] for split in variable.keys()])
    variable = variable.remove_columns([col for col in variable.column_names if col != col_to_keep])
    if col_to_keep != "text":
        variable = variable.rename_column(col_to_keep, "text")
    variable = variable.map(lambda x: {"label": label}, num_proc=7)
    return variable

reddit = datasetloader("SophieTr/reddit_clean", "content", 1)
reddit = concatenate_datasets([reddit, datasetloader("solomonk/reddit_mental_health_posts", "body", 1)])
#reddit = concatenate_datasets([reddit, datasetloader("sentence-transformers/reddit", "body", 1)])
reddit = concatenate_datasets([reddit, datasetloader("winddude/reddit_finance_43_250k", "selftext", 1)])
#This one has another column could use the data of
#TODO get list of answaers
#reddit = concatenate_datasets([reddit, datasetloader("nreimers/reddit_question_best_answers", "body", 1)])

#twitter data loads
#twitter = datasetloader()


combined_ds = concatenate_datasets([fourchannel, reddit])
combined_ds = combined_ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0, num_proc=7)
combined_ds = combined_ds.train_test_split(test_size=0.2, seed=42)
print(combined_ds.shape)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(inp):
    # inp["text"] is a list of strings when batched=True
    tokenized = tokenizer(
        inp["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    # Return the tokenized outputs directly
    # This will add input_ids, attention_mask, etc. as new columns
    return tokenized

tokenized = combined_ds.map(tokenize, batched=True, num_proc=7, batch_size=500)
print("DONE TOKENIZING")

#labels = dataset["train"]["label"]
#TODO this  vvvv
#class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
#print(class_weights)

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Keep only the classification head trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    auto_find_batch_size=True,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
    eval_strategy="epoch",      # Evaluate at the end of each epoch
    save_strategy="epoch",            # Save at the end of each epoch
)
print(training_args)
print(model.config)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load a metric (F1-score in this case)
metric = load("f1")

# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./my_trained_model")
tokenizer.save_pretrained("./my_trained_model")

print("Model saved to ./my_trained_model")
