import copy
import datetime
import os
import re
import pandas as pd
import time
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

import sys
sys.path.insert(-1,'')
from test import test_nlp

seed = 42
torch.manual_seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)  
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits

test_interval_steps = 100

def nlp_learn(model, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, num_epochs=3, lr=5e-6, weight_decay=0.01, T=1, lambda_=0.2):
    import torch

    old_model = copy.deepcopy(model)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(cl_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            soft_labels = nn.functional.softmax(old_outputs / T, dim=1)
            if new_class_num > old_classes_num:
                new_soft_labels = torch.zeros(soft_labels.size(0), new_class_num).to(device)
                new_soft_labels[:, :old_classes_num] = soft_labels
                for i in range(old_classes_num, new_class_num):
                    new_soft_labels[:, i] = 1e-9
            else:
                new_soft_labels = soft_labels

            distillation_loss = -torch.mean(torch.sum(new_soft_labels * nn.functional.log_softmax(outputs / T, dim=1), dim=1)) * (T*T)

            total_loss = (1 - lambda_) * loss + lambda_ * distillation_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=total_loss.item())
            epoch_loss += total_loss.item()

            if step % test_interval_steps == 0:
                test_start_time = time.time()

                elapsed_training_time = time.time() - start_time - total_test_time
                unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                retain_acc = test_nlp(model, test_dataloader)
                log_entry = {
                    "step": step,
                    "training_time": elapsed_training_time,
                    "unlearn_accuracy": unlearn_acc,
                    "retain_accuracy": retain_acc
                }
                accuracy_log.append(log_entry)
                print(f"Logged accuracy: {log_entry}")
                total_test_time += (time.time() - test_start_time)
                model.train()
        
        avg_loss = epoch_loss / (progress_bar.n + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)
    
    return model, accuracy_log

def nlp_learn_nodistill(model, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, num_epochs=3, lr=5e-6, weight_decay=0.01, T=1, lambda_=0.2):
    import torch

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(cl_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

            if step % test_interval_steps == 0:
                test_start_time = time.time()

                elapsed_training_time = time.time() - start_time - total_test_time
                unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                retain_acc = test_nlp(model, test_dataloader)
                log_entry = {
                    "step": step,
                    "training_time": elapsed_training_time,
                    "unlearn_accuracy": unlearn_acc,
                    "retain_accuracy": retain_acc
                }
                accuracy_log.append(log_entry)
                print(f"Logged accuracy: {log_entry}")
                total_test_time += (time.time() - test_start_time)
                model.train()
        
        avg_loss = epoch_loss / (progress_bar.n + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)
    
    return model, accuracy_log

def nlp_unlearn(model, ul_dataloader, test_dataloader, test_unlearn_dataloader, num_epochs=3, lr=2e-7, alpha=0.8, beta=0.1, T=1):
    import torch
    old_model = copy.deepcopy(model)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    offset = 0.01
    epsilon = 1e-9
    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(ul_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)

            soft_labels = nn.functional.softmax(old_outputs / T, dim=1)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = nn.functional.softmax(outputs, dim=1)

            reduction_loss = -torch.log(1 - (logits[range(input_ids.size(0)), labels] - offset)).mean()
            perplexity_loss = -torch.sum(-logits * torch.log(logits + epsilon), dim=1).mean()

            log_probs = nn.functional.log_softmax(outputs / T, dim=1)
            kl_loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

            loss = (1 - alpha) * reduction_loss + alpha * perplexity_loss - beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

            if step % test_interval_steps == 0:
                test_start_time = time.time()

                elapsed_training_time = time.time() - start_time - total_test_time
                unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                retain_acc = test_nlp(model, test_dataloader)
                log_entry = {
                    "step": step,
                    "training_time": elapsed_training_time,
                    "unlearn_accuracy": unlearn_acc,
                    "retain_accuracy": retain_acc
                }
                accuracy_log.append(log_entry)
                print(f"Logged accuracy: {log_entry}")
                total_test_time += (time.time() - test_start_time)
                model.train()
            
        avg_loss = epoch_loss / (progress_bar.n + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)
    
    return model, accuracy_log

def nlp_unlearn_neggrad(model, ul_dataloader, test_dataloader, test_unlearn_dataloader, num_epochs=3, lr=2e-7, alpha=0.8, beta=0.1, T=1):
    import torch
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    offset = 0.01
    epsilon = 1e-9
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(ul_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in progress_bar:
            step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = -1 * criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

            if step % test_interval_steps == 0:
                test_start_time = time.time()

                elapsed_training_time = time.time() - start_time - total_test_time
                unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                retain_acc = test_nlp(model, test_dataloader)
                log_entry = {
                    "step": step,
                    "training_time": elapsed_training_time,
                    "unlearn_accuracy": unlearn_acc,
                    "retain_accuracy": retain_acc
                }
                accuracy_log.append(log_entry)
                print(f"Logged accuracy: {log_entry}")
                total_test_time += (time.time() - test_start_time)
                model.train()
            
        avg_loss = epoch_loss / (progress_bar.n + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)
    
    return model, accuracy_log

def nlp_alternate_learn_unlearn(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)

                reduction_loss = -torch.log(1 - (logits[range(input_ids.size(0)), labels] - offset)).mean()
                perplexity_loss = -torch.sum(-logits * torch.log(logits + epsilon), dim=1).mean()

                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                kl_loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

                loss = (1 - alpha) * reduction_loss + alpha * perplexity_loss - beta * kl_loss
                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                distillation_loss = -torch.mean(torch.sum(soft_labels * nn.functional.log_softmax(outputs / T, dim=1), dim=1)) * (T*T)

                total_loss = (1 - lambda_) * loss + lambda_ * distillation_loss

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log

def tab_alternate_learn_unlearn(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)

                reduction_loss = -torch.log(torch.clamp(1 - (logits[range(input_ids.size(0)), labels] - offset), min=1e-9)).mean()

                perplexity_loss = -torch.sum(-logits * torch.log(torch.clamp(logits, min=epsilon)), dim=1).mean()

                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                soft_labels = torch.clamp(soft_labels, min=1e-9)
                kl_loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

                loss = (1 - alpha) * reduction_loss + alpha * perplexity_loss - beta * kl_loss
                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                #print(loss.item())

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                distillation_loss = -torch.mean(torch.sum(soft_labels * nn.functional.log_softmax(outputs / T, dim=1), dim=1)) * (T*T)

                total_loss = (1 - lambda_) * loss + lambda_ * distillation_loss

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                #print(total_loss.item())

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log


def nlp_alternate_learn_unlearn_0(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    """
    hard+hard
    """
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = -1 * criterion(logits, labels)

                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                total_loss = criterion(logits, labels)

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log


def nlp_alternate_learn_unlearn_1(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    """
    Ours+hard
    """
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = -1 * criterion(logits, labels)

                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                soft_labels = nn.functional.softmax(old_outputs / T, dim=1)
                if new_class_num > old_classes_num:
                    new_soft_labels = torch.zeros(soft_labels.size(0), new_class_num).to(device)
                    new_soft_labels[:, :old_classes_num] = soft_labels
                    for i in range(old_classes_num, new_class_num):
                        new_soft_labels[:, i] = 1e-9
                else:
                    new_soft_labels = soft_labels

                distillation_loss = -torch.mean(torch.sum(new_soft_labels * nn.functional.log_softmax(outputs / T, dim=1), dim=1)) * (T*T)

                total_loss = (1 - lambda_) * loss + lambda_ * distillation_loss

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log


def nlp_alternate_learn_unlearn_2(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    """
    hard+ours
    """
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)

                reduction_loss = -torch.log(1 - (logits[range(input_ids.size(0)), labels] - offset)).mean()
                perplexity_loss = -torch.sum(-logits * torch.log(logits + epsilon), dim=1).mean()

                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                kl_loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

                loss = (1 - alpha) * reduction_loss + alpha * perplexity_loss - beta * kl_loss

                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                total_loss = criterion(logits, labels)

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log


def nlp_alternate_learn_unlearn_3(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    """
    hard+soft
    """
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)

                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                total_loss = criterion(logits, labels)

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log


def nlp_alternate_learn_unlearn_4(model, ul_dataloader, cl_dataloader, test_dataloader, test_unlearn_dataloader, new_class_num, 
                                unlearn_epochs=3, learn_epochs=3, unlearn_lr=2e-7, learn_lr=5e-6, 
                                weight_decay=0.01, alpha=0.8, beta=0.1, T=1, lambda_=0.2):
    """
    ours+soft
    """
    import torch
    from tqdm import tqdm

    offset = 0.01
    epsilon = 1e-9

    old_model = copy.deepcopy(model)
    unlearn_optimizer = AdamW(model.parameters(), lr=unlearn_lr)
    learn_optimizer = AdamW(model.parameters(), lr=learn_lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    old_classes_num = model.classifier.out_features
    print("old_class:{}, new_class:{}".format(old_classes_num, new_class_num))
    if new_class_num > old_classes_num:
        # Expand the classifier for the new model
        old_weight = model.classifier.weight.data
        old_bias = model.classifier.bias.data
        hidden_size = old_weight.size(1)
        new_classifier = torch.nn.Linear(hidden_size, new_class_num)
        with torch.no_grad():
            new_classifier.weight[:old_classes_num] = old_weight
            new_classifier.bias[:old_classes_num] = old_bias
        model.classifier = new_classifier

    old_model.to(device)
    model.to(device)

    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []

    # Calculate total steps for unlearn and learn
    unlearn_total_steps = unlearn_epochs * len(ul_dataloader)
    learn_total_steps = learn_epochs * len(cl_dataloader)
    print("Unlearn steps:{}, Learn steps:{}.".format(unlearn_total_steps, learn_total_steps))

    # Calculate 10% steps for unlearn and learn
    unlearn_batch_size = max(1, int(0.1 * unlearn_total_steps))
    learn_batch_size = max(1, int(0.1 * learn_total_steps))

    # Initialize dataloader iterators
    unlearn_iter = iter(ul_dataloader)
    learn_iter = iter(cl_dataloader)

    # Log initial accuracy
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": 0,
        "training_time": 0,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    old_model.eval()
    model.train()
    start_time = time.time()

    # Alternate between unlearn and learn in 10% batches
    while unlearn_total_steps > 0 or learn_total_steps > 0:
        # Perform 10% unlearn steps
        if unlearn_total_steps > 0:
            for _ in range(unlearn_batch_size):
                try:
                    batch = next(unlearn_iter)
                except StopIteration:
                    unlearn_iter = iter(ul_dataloader)
                    batch = next(unlearn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)

                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean', log_target=False) * (T ** 2)

                unlearn_optimizer.zero_grad()
                loss.backward()
                unlearn_optimizer.step()

                step += 1
                unlearn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

        # Perform 10% learn steps
        if learn_total_steps > 0:
            for _ in range(learn_batch_size):
                try:
                    batch = next(learn_iter)
                except StopIteration:
                    learn_iter = iter(cl_dataloader)
                    batch = next(learn_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with torch.no_grad():
                    old_outputs = old_model(input_ids=input_ids, attention_mask=attention_mask)
                    old_outputs = old_outputs.logits if hasattr(old_outputs, 'logits') else old_outputs

                # Handle the case where old_outputs and new_outputs have different dimensions
                if new_class_num > old_classes_num:
                    # Expand old_outputs to match the new_class_num
                    expanded_old_outputs = torch.zeros(old_outputs.size(0), new_class_num).to(device)
                    expanded_old_outputs[:, :old_classes_num] = old_outputs
                    soft_labels = nn.functional.softmax(expanded_old_outputs / T, dim=1)
                else:
                    soft_labels = nn.functional.softmax(old_outputs / T, dim=1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits if hasattr(outputs, 'logits') else nn.functional.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                distillation_loss = -torch.mean(torch.sum(soft_labels * nn.functional.log_softmax(outputs / T, dim=1), dim=1)) * (T*T)

                total_loss = (1 - lambda_) * loss + lambda_ * distillation_loss

                learn_optimizer.zero_grad()
                total_loss.backward()
                learn_optimizer.step()

                step += 1
                learn_total_steps -= 1

                # Test every test_interval_steps (based on total steps)
                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

    # Final test and log
    total_training_time = time.time() - start_time - total_test_time
    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
    retain_acc = test_nlp(model, test_dataloader)
    log_entry = {
        "step": step,
        "training_time": total_training_time,
        "unlearn_accuracy": unlearn_acc,
        "retain_accuracy": retain_acc
    }
    accuracy_log.append(log_entry)

    return model, accuracy_log