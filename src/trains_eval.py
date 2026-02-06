import torch.nn as nn
import time
import torch
import sklearn.metrics import accuracy_score

# loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reusable component generates loss and logits from embedding and classifier
def compute_logits_loss(bert, model, criterion, input_ids, attention_mask, labels):
    outputs = bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = model(outputs.last_hidden_state, outputs.pooler_output)
    loss = criterion(logits, labels)

    return logits, loss


def train(bert, model, criterion, optimizer, dataloader):
    model.to(device)
    bert.to(device)

    model.train()
    bert.train()

    total_loss, correct = 0, 0
    for _, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        logits, loss = compute_logits_loss(
            bert, model, criterion, input_ids, attention_mask, labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predict = torch.argmax(logits, dim=1)
        correct += accuracy_score(predict, labels)

def eval(bert, model, criterion, dataloader):
    model.to(device)
    bert.to(device)

    model.eval()
    bert.eval()

    all_preds, true_labels = [], []
    total_loss, total = 0, 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            inputs, attention_mask,labels = batch
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            logits, loss = compute_logits_loss(bert, model, criterion, inputs, attention_mask, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            total += labels.size(0)

            all_preds.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    return total_loss / total, all_preds, true_labels
