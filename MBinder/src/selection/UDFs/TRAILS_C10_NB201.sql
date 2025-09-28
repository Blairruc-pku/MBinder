CREATE OR REPLACE PROCEDURE TRAILS_C10_NB201(target_table TEXT, sample_num INTEGER)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    This is used for comparison.
    TRAILS selects model with 2 phases.
    Phase 1: filter model with trainability and express ability
    Phase 2: filter model with SH.
    """
    import base64
    import io
    import os
    import time
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm

    import sys

    from xautodl.models import get_cell_based_tiny_net

    from NAS_Dataset import CustomDataset

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    # Load Model files
    model_repo_dir = ''
    model_dirs = [d for d in os.listdir(model_repo_dir) if os.path.isdir(os.path.join(model_repo_dir, d))]
    config_path_list = [os.path.join(model_repo_dir, model_dir, "config.json") for model_dir in model_dirs]

    # Load target table data
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT image, label FROM ( \
    SELECT encode(image, 'base64') as image, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= {};".format(target_table, str(sample_num))
    TARGET_TABLE = plpy.execute(query)
    for data in TARGET_TABLE:
        decoded_image = base64.b64decode(data['image'])
        data['image'] = Image.open(io.BytesIO(decoded_image)).convert("RGB")
        data['label'] = int(data['label'])

    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)[0]['count']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loss_fn = nn.CrossEntropyLoss()

    print("----------TRAILS PHASE 1----------")

    data_by_label = {label: [] for label in range(label_num)} 
    for row in TARGET_TABLE:
        data_by_label[row["label"]].append(row["image"])

    alpha = 0.8
    model_score = {}

    for i, model_dir in enumerate(model_dirs):
        config_path = config_path_list[i]
        with open(config_path, "r") as f:
            config = json.load(f)
        config['num_classes'] = label_num
        if label_num>64:
            config['C'] = 32
        model = get_cell_based_tiny_net(config).to(device)
        model.eval()

        losses = []
        log_k_h_values = []

        for idx in range(sample_num):
            selected_image = [data_by_label[label][idx] for label in data_by_label]
            selected_labels = list(data_by_label.keys())

            inputs = torch.stack([transform(img) for img in selected_image]).to(device)
            labels_tensor = torch.tensor(selected_labels).to(device)

            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs[0]
                loss = loss_fn(logits, labels_tensor)

                hidden_states = outputs[1]
                cls_activations = hidden_states
                relu_activations = torch.relu(cls_activations)
                binary_pattern = (relu_activations > 0).int()

                num_samples = len(selected_image)
                K_H = torch.zeros((num_samples, num_samples))
                for i in range(num_samples):
                    for j in range(i, num_samples):
                        hamming_dist = (binary_pattern[i] != binary_pattern[j]).sum().item()
                        K_H[i, j] = K_H[j, i] = binary_pattern.shape[1] - hamming_dist  # N_a = hidden_dim

                det_k_h = torch.linalg.det(K_H)
                log_K_H = torch.log2(det_k_h).item()

            losses.append(loss.cpu().item())
            log_k_h_values.append(log_K_H)

        average_loss = np.mean(losses)
        average_log_k_h = np.mean(log_k_h_values)
        proxy_score = alpha * average_loss + (1 - alpha) * average_log_k_h

        model_score[model_dir] = proxy_score

        model_path = model_repo_dir+"/"+model_dir+"/epoch0"
        print(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model, model_path+"/pytorch_model.pth")
        del model, losses, log_k_h_values

    model_score = sorted(model_score.items(), key=lambda kv: (kv[1], kv[0]))
    deleted_model = [item[0] for item in model_score[:len(model_score)-100]]
    model_dirs = [item for item in model_dirs if item not in deleted_model]

    print("----------TRAILS PHASE 2----------")

    temp = {'img': [], 'label': []}
    for item in TARGET_TABLE:
        temp['img'].append(item['image'])
        temp['label'].append(item['label'])
    df = pd.DataFrame(temp)

    train_data = []
    val_data = []

    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        label_data = label_data.reset_index(drop=True)

        train_data.append(label_data.iloc[:int(sample_num*0.8)])
        val_data.append(label_data.iloc[int(sample_num*0.8):sample_num])

    train_data = pd.concat(train_data).reset_index(drop=True)
    val_data = pd.concat(val_data).reset_index(drop=True)

    train_data = train_data.to_dict('records')
    val_data = val_data.to_dict('records')

    del temp, df
    train_dataset = CustomDataset(train_data, transform=transform)
    val_dataset = CustomDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    epoch = 2
    args = {'learn_rate': 1e-3, 'epochs': 2}

    while len(model_dirs) > 1:
        print("----------epoch {}----------".format(epoch))
        epoch_acc = {}
        for i, model_dir in enumerate(model_dirs):
            model = torch.load(f"{model_repo_dir}/{model_dir}/epoch{epoch-2}/pytorch_model.pth").to(device)

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learn_rate'])
            for _ in range(args['epochs']):
                model.train()
                for images, labels in tqdm(train_loader, desc='Training:'):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs[0], labels)
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Validating'):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    predicted = outputs[0].argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            epoch_acc[model_dir] = accuracy
            print("model:{}, acc:{}".format(model_dir, accuracy))

            # Save the model
            output_path = f"{model_repo_dir}/{model_dir}/epoch{epoch}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            torch.save(model, output_path+"/pytorch_model.pth")

        # Select top half of models
        epoch_acc = sorted(epoch_acc.items(), key=lambda kv: (kv[1], kv[0]))
        model_dirs = [item[0] for item in epoch_acc[len(epoch_acc)//2:]]

        epoch += 2

    print("Model {} selected.".format(model_dirs[0]))
    end_time = time.time()
    log = []
    item = {
        "time": end_time - start_time,
        "info": model_dirs[0]
    }
    log.append(item)

$$;