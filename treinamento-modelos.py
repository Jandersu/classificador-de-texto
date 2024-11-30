import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np
import pandas as pd

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_LABELS = 2
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5

tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_roberta = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

base_model_bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
base_model_roberta = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

class Classifier(nn.Module):
    def __init__(self, model, num_labels):
        super(Classifier, self).__init__()
        self.base_model = model
        self.cls = nn.Linear(model.config.hidden_size, 400)
        self.dropout = nn.Dropout(p=0.5)
        self.cls2 = nn.Linear(400, num_labels)
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0][:, 0, :]  # CLS token output
        prediction = self.cls(sequence_output)
        prediction = self.gelu(prediction)
        prediction = self.dropout(prediction)
        prediction = self.cls2(prediction)
        return prediction

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        label = torch.tensor(self.labels[idx])
        return item, label

    def __len__(self):
        return len(self.labels)

def train(model, train_loader, optimizer, loss_fct):
    model.train()
    epoch_losses = []
    for batch, labels in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fct(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    return np.mean(epoch_losses)

def evaluate(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs, dim=-1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
    return y_true, y_pred

def load_dataset(csv_file, sample_size=None):
    data = pd.read_csv(csv_file)

    texts = data['text'].tolist()
    labels = data['label'].tolist()

    xtrain_global = np.array(texts)
    ytrain_global = np.array(labels)

    return xtrain_global, ytrain_global

def main():
    xtrain_global, ytrain_global = load_dataset("dataset.csv")

    xtrain, xval, ytrain, yval = train_test_split(
        xtrain_global, ytrain_global, test_size=0.30, random_state=42, shuffle=True
    )

    #tokenizer bert
    train_encodings_bert = tokenizer_bert(xtrain.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings_bert = tokenizer_bert(xval.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

    #tokenizer roberta
    train_encodings_roberta = tokenizer_roberta(xtrain.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings_roberta = tokenizer_roberta(xval.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

    #datasets e loader da bert
    train_dataset_bert = MyDataset(train_encodings_bert, ytrain)
    val_dataset_bert = MyDataset(val_encodings_bert, yval)
    train_loader_bert = DataLoader(train_dataset_bert, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bert = DataLoader(val_dataset_bert, batch_size=BATCH_SIZE)

    #datasets e loader da roberta
    train_dataset_roberta = MyDataset(train_encodings_roberta, ytrain)
    val_dataset_roberta = MyDataset(val_encodings_roberta, yval)
    train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=BATCH_SIZE)

    model_bert = Classifier(base_model_bert, NUM_LABELS).to(DEVICE)
    model_roberta = Classifier(base_model_roberta, NUM_LABELS).to(DEVICE)

    optimizer_bert = AdamW(model_bert.parameters(), lr=LEARNING_RATE)
    optimizer_roberta = AdamW(model_roberta.parameters(), lr=LEARNING_RATE)

    loss_fct = nn.CrossEntropyLoss()

    #treino bert
    for epoch in range(NUM_EPOCHS):
        train_loss_bert = train(model_bert, train_loader_bert, optimizer_bert, loss_fct)
        print(f"[BERT] Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {train_loss_bert:.4f}")

    y_true_bert, y_pred_bert = evaluate(model_bert, val_loader_bert)
    print("[BERT] Classification Report:")
    print(metrics.classification_report(y_true_bert, y_pred_bert))

    #treino roberta
    for epoch in range(NUM_EPOCHS):
        train_loss_roberta = train(model_roberta, train_loader_roberta, optimizer_roberta, loss_fct)
        print(f"[Roberta] Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {train_loss_roberta:.4f}")

    y_true_roberta, y_pred_roberta = evaluate(model_roberta, val_loader_roberta)
    print("[Roberta] Classification Report:")
    print(metrics.classification_report(y_true_roberta, y_pred_roberta))

    #salvar modelos
    torch.save(model_bert.state_dict(), "model-bert.pth")
    torch.save(model_roberta.state_dict(), "model-roberta.pth")
    print("Models saved as 'model-bert.pth' and 'model-roberta.pth'.")

if __name__ == '__main__':
    main()
