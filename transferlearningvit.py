import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
import os

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print ("MPS device found.")
    else:
        device = 'cpu'
        print ("MPS device not found.")
    return device

# Define a new classifier on top of the pre-trained encoder
class ViTClassifier(nn.Module):
    def __init__(self, vit_model, num_classes, config):
        super(ViTClassifier, self).__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use the CLS token output
        logits = self.classifier(cls_output)
        return logits

# Define EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoints.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_dirs(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def transfer_learning_pretrain_vit(model_checkpoint, train_loader, val_loader, test_loader, dataset, num_labels_ = 120,
                                   learning_rate = 5e-5, num_epochs = 10, patience_ = 3):
    device = get_device()
    subfolder_name = model_checkpoint.split('/')[1]
    create_dirs(f'results/transferlearning/{dataset}/{subfolder_name}')
    create_dirs(f'checkpoints/transferlearning/{dataset}')

    # Load the pre-trained ViT model
    config = AutoConfig.from_pretrained(model_checkpoint)
    pretrained_model = AutoModel.from_pretrained(model_checkpoint)

    # vit_model = ViTModel.from_pretrained(model_)
    pretrained_model.to(device)

    # Freeze the pre-trained model parameters
    for param in pretrained_model.parameters():
        param.requires_grad = False

    model = ViTClassifier(pretrained_model, num_classes=num_labels_, config = config)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=patience_, verbose=True, path = f'checkpoints/transferlearning/{dataset}/{subfolder_name}.pt')

    # Metrics storage
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load(early_stopping.path))

    # Test phase
    model.eval()
    test_running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_running_loss / len(test_loader)
    test_loss_list.append(avg_test_loss)

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

    # Save metrics to a CSV file
    metrics = {
        'Train Loss': train_loss_list,
        'Validation Loss': val_loss_list,
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1 Score': f1_list
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'results/transferlearning/{dataset}/{subfolder_name}/training_metrics.csv', index=False)

    # Sample list of test metrics
    numbers = [test_accuracy, test_precision, test_recall, test_f1]

    # Create a DataFrame from the list
    df = pd.DataFrame(numbers, columns=['Numbers'])

    # Save the DataFrame to a CSV file
    df.to_csv(f'results/transferlearning/{dataset}/{subfolder_name}/test_metrics.csv', index=False)