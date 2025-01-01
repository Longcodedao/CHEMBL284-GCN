from sklearn.model_selection import train_test_split
import torch


# Split the dataset into train, validation, and test
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    train_data, temp_data = train_test_split(dataset, test_size=1 - train_ratio,
                                             random_state=random_seed)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)
    return train_data, val_data, test_data


def train_batch(train_loader, model, optimizer, loss_fn, 
                scheduler = None, device = 'cuda', clip_value = 1.0, display = True):
    
    running_loss = 0.0


    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        node_mat = batch['node_mat'].to(device)
        adj_mat = batch['adj_mat'].to(device)
        output = batch['output'].to(device)

        pred = model(node_mat, adj_mat)
        loss = loss_fn(pred, output)
        loss.backward()

        if display and idx % 100 == 1:
            print(f'[Batch {idx}/{len(train_loader)}] Loss: {loss.item():.4f}')  

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
     
    return running_loss / len(train_loader)


def evaluate_batch(eval_loader, model, loss_fn, device = 'cuda'): 

    running_loss = 0
    with torch.no_grad():

        model.eval()

        for batch in eval_loader:
            node_mat = batch['node_mat'].to(device)
            adj_mat = batch['adj_mat'].to(device)
            output = batch['output'].to(device)

            pred = model(node_mat, adj_mat)
            loss = loss_fn(pred, output)
            running_loss += loss.item()


        val_loss = running_loss / len(eval_loader)

    return val_loss