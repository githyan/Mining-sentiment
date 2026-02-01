import torch.nn as nn
import time
import torch


loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reusable component generates loss and logits from embedding and classifier
def compute_logits_loss(bert, model, input_ids, attention_mask, labels):
    outputs = bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
    )
    logits = model(outputs.last_hidden_state, outputs.pooler_output)
    loss = loss_fn(logits, labels)

    return logits, loss
    

# Calculates loss and logits 
def getPredicts(bert, model, optimizer, train_dataloader, is_train=True):
    total_loss, correct, total = 0, 0,0
    # make the batch dataloader
    for i, batching in enumerate(train_dataloader):
        # Connect the token embedding to device
        input_ids, attention_mask, labels = batching
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # This section will be trainable if it's True returns zero grad to accumulate the predictions
        # False would become model.eval() just for validation dataset exists for handling overfitting
        if is_train:
            accumulation_step = 4
            optimizer.zero_grad()
            logits, loss = compute_logits_loss(bert, model, input_ids, attention_mask, labels)
            loss = loss / accumulation_step
            loss.backward()
            optimizer.step()

            # Optimize every accumulation step from batching dataset
            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            logits, loss = compute_logits_loss(bert, model, input_ids, attention_mask, labels)

        # compute loss and accuracy 
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = total_loss / total
    epoch_train_acc = 100 * correct / total

    return epoch_train_loss, epoch_train_acc

def train(bert, model, optimizer, train_dataloader, val_dataloader=None, epochs=2):
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(epochs):
        t0 = time.time()
        bert.train()
        model.train()

        train_loss, train_acc = getPredicts(bert, model, optimizer, train_dataloader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if val_dataloader:
            bert.eval()
            model.eval()
            
            with torch.no_grad():
                val_loss, val_acc = getPredicts(bert, model, optimizer, val_dataloader, is_train=False)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            print(f'Epochs: {epoch + 1} | Train Loss: {train_loss: .2f} | Train Acc: {train_acc: .4f} | Val Loss: {val_loss: .2f} | Val Acc: {val_acc: .4f} Times: {time.time() - t0: .2f}s')

        else:
            print(f'Epochs: {epoch + 1} | Train Loss: {train_loss: .2f} | Train Acc: {train_acc: .4f} | Times: {time.time() - t0: .2f}s')
