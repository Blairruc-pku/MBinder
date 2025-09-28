CREATE OR REPLACE PROCEDURE TRAILS_DNN_Frappe(target_table TEXT, sample_num INTEGER)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    This is used for comparison.
    TRAILS selects model with 2 phases.
    Phase 1: filter model with trainability and express ability
    Phase 2: filter model with SH.
    """
    import re
    import os
    import time
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    import sys
    class TabularDNN(nn.Module):
        def __init__(self, input_dim=10, hidden_sizes=[64, 32, 16, 8]):
            super(TabularDNN, self).__init__()
            assert len(hidden_sizes) == 4, 
            
            self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
            self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
            self.output = nn.Linear(hidden_sizes[3], 2)
            
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.output(x)
            return x

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    # Load Model files
    model_repo_dir = ''
    model_files = [f for f in os.listdir(model_repo_dir) if f.endswith('.pth')]

    # Load target table data
    plpy.execute("SELECT setseed(0.42)")
    query = f"""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = '{target_table.lower()}' 
    AND column_name != 'label'
    ORDER BY ordinal_position
    """
    columns_result = plpy.execute(query)
    feature_columns = [row['column_name'] for row in columns_result]
    query = f"""
    SELECT {', '.join(feature_columns)}, label FROM (
        SELECT {', '.join(feature_columns)}, label, 
               ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn 
        FROM {target_table}
    ) AS subquery WHERE rn <= {sample_num};
    """
    result = plpy.execute(query)
    
    # Prepare data
    features = []
    labels = []
    for row in result:
        features.append([row[col] for col in feature_columns])
        labels.append(row['label'])

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    print("----------TRAILS PHASE 1----------")

    unique_labels = np.unique(labels)
    data_by_label = {label: features[labels == label] for label in unique_labels}
    
    alpha = 0.8
    model_score = {}

    for model_file in tqdm(model_files, desc="Evaluating models in Phase 1"):
        # Load model
        sizes = list(map(int, re.findall(r'dnn_(\d+)_(\d+)_(\d+)_(\d+)\.pth', model_file)[0]))
        model = TabularDNN(hidden_sizes=sizes).to(device)
        model_path = os.path.join(model_repo_dir, model_file)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        losses = []
        log_k_h_values = []

        for idx in range(sample_num):
            selected_image = [data_by_label[label][idx] for label in data_by_label]
            selected_labels = list(data_by_label.keys())

            selected_features = [data_by_label[label][idx % len(data_by_label[label])] 
                                for label in data_by_label]
            selected_labels = list(data_by_label.keys())
            
            inputs = torch.tensor(selected_features, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(selected_labels).to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels_tensor)
                hidden_activations = []
                def hook(module, input, output):
                    hidden_activations.append(output)
                
                handle = model.fc4.register_forward_hook(hook)
                model(inputs)
                handle.remove()
                
                activations = hidden_activations[0]
                binary_pattern = (activations > 0).int()

                num_samples = len(selected_image)
                K_H = torch.zeros((num_samples, num_samples), device=device)
                for i in range(num_samples):
                    for j in range(i, num_samples):
                        hamming_dist = (binary_pattern[i] != binary_pattern[j]).sum().item()
                        K_H[i, j] = K_H[j, i] = binary_pattern.shape[1] - hamming_dist  # N_a = hidden_dim

                det_k_h = torch.linalg.det(K_H + torch.eye(num_samples, device=device) * 1e-3)
                log_K_H = torch.log2(det_k_h).item()

            losses.append(loss.cpu().item())
            log_k_h_values.append(log_K_H)

        average_loss = np.mean(losses)
        average_log_k_h = np.mean(log_k_h_values)
        proxy_score = alpha * average_loss + (1 - alpha) * average_log_k_h

        model_score[model_file] = proxy_score

    model_score = sorted(model_score.items(), key=lambda kv: (kv[1], kv[0]))
    selected_models = [item[0] for item in model_score[-100:]]

    print("----------TRAILS PHASE 2----------")

    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split = int(0.8 * len(features))
    train_idx, val_idx = indices[:split], indices[split:]

    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)

    train_features = features_tensor[train_idx]
    train_labels = labels_tensor[train_idx]
    val_features = features_tensor[val_idx]
    val_labels = labels_tensor[val_idx]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    epoch = 1
    args = {'learn_rate': 1e-4, 'epochs': 1}

    while len(selected_models) > 1:
        print("----------epoch {}----------".format(epoch))
        epoch_acc = {}
        for model_file in tqdm(selected_models, desc="Evaluating models in Phase 1"):
            sizes = list(map(int, re.findall(r'dnn_(\d+)_(\d+)_(\d+)_(\d+)\.pth', model_file)[0]))
            model = TabularDNN(hidden_sizes=sizes).to(device)
            model_path = os.path.join(model_repo_dir, model_file)
            model.load_state_dict(torch.load(model_path))

            # Training loop
            optimizer = torch.optim.AdamW(model.parameters(), lr=args['learn_rate'])
            for _ in range(args['epochs']):
                model.train()
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = loss_fn(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    predicted = outputs.argmax(dim=1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            accuracy = 100 * correct / total
            epoch_acc[model_file] = accuracy
            print("model:{}, acc:{}".format(model_file, accuracy))

            torch.save(model.state_dict(), model_file)

        # Select top half of models
        epoch_acc = sorted(epoch_acc.items(), key=lambda kv: (kv[1], kv[0]))
        selected_models = [item[0] for item in epoch_acc[len(epoch_acc)//2:]]

        epoch += 2

    print("Model {} selected.".format(selected_models[0]))
    end_time = time.time()
    log = []
    item = {
        "time": end_time - start_time,
        "info": selected_models[0]
    }
    log.append(item)

$$;