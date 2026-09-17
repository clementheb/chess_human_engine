import torch.nn as nn, torch.nn.functional as F, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import model.model_cfg as cfg

from data_process.dataset_builder import LichessGames
from data_process.data_encoder import decode_board, legal_moves
from sklearn.metrics import accuracy_score
import os
import time
import mlflow

from scripts.helpers import extract_topk, topk_accuracy



class CNN_base(nn.Module):
    def __init__(self):
        super(CNN_base,self).__init__()
        ##definir les couches/sequences du NN. 
        # Possible de juste donner un nom à une couche mais surtout à des blocs. Ex mon ResBlock.
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=cfg.in_chan, out_channels= 32, kernel_size=3,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=4,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=4,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=4,stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05)
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=5*5*16, out_features=256),
            nn.Tanh(),
            nn.Linear(256, 4096),
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        return(x)  


class CNN_resblock(nn.Module):
    def __init__(self):
        super(CNN_resblock,self).__init__()
        ##definir les couches/sequences du NN. 
        # Possible de juste donner un nom à une couche mais surtout à des blocs. Ex mon ResBlock.
        self.first_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=cfg.in_chan, out_channels= 32, kernel_size=3,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=4,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=4,stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=4,stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05)
        )

        self.conv_block = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=5*5*16, out_features=256),
            nn.Tanh(),
            nn.Linear(256, 4096),
        )
        
    def forward(self, x):
        x = self.first_conv_block(x)
        x = x + self.conv_block(x) ## Resblock right ?
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        return(x)  



def train_model(data, model):
    """
    Input data as instance of class of your dataset
    Model should be input of instance of model class
    Trains model and saves it in model path specified in model_cfg.py"""

    eta = cfg.eta
    EPOCHS = cfg.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used for PyTorch computing :", device)
    start_epc = 0

    train_params = {
        "eta": eta,
        "epochs": EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
    }
    
    with mlflow.start_run():

        mlflow.log_params(train_params)
        model.to(device)

        data.mode = 'train'
        optimizer = torch.optim.AdamW(model.parameters(), lr=eta)
        scheduler = CosineAnnealingLR(optimizer=optimizer, eta_min = eta/50, T_max=cfg.epochs)
        train_data = DataLoader(data, batch_size=cfg.BATCH_SIZE, shuffle=True)

        if cfg.RESUME:
            checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epc = checkpoint['epoch']

        model.train()
        start_time = time.time()   

        for epoch in range(start_epc+1, EPOCHS+1):
            epoch_start = time.time()
            losses = []
            for D in train_data:
                optimizer.zero_grad()
                data = D['position'].to(device)
                label = D['move'].to(device)
                y_hat = model(data)
                error = nn.CrossEntropyLoss()
                loss = error(y_hat,label).sum()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            loss = float(np.mean(losses))

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            print(f"\n Epoch {epoch}/{EPOCHS-1} | "
                f"Epoch time: {epoch_time:.2f}s | "
                f"Total elapsed: {total_time/60:.2f} min | "
                f"Loss: {loss:.4f}")
            mlflow.log_metrics({"train_loss": loss}, step=epoch)

        os.makedirs(cfg.MODEL_SAVE_FOLDER, exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }, cfg.MODEL_SAVE_PATH)


def eval_model(dataset, model):
    dataset.mode = 'test'
    dataloader = DataLoader(dataset,batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, weights_only=True))
    model.eval() #definit un mode pour le modele, pour les opérations à venir

    output = []
    legal_output = []
    y_true = []

    with torch.no_grad():
        sample_index = 0
        print("Evaluating model")
        for D in dataloader:
            move = D["move"].to(device)
            position = D['position'].to(device, dtype = torch.float32)
            batch_out = model(position) #batch of logits output
            batch_logits = torch.softmax(batch_out, dim=1)

            np_logits = batch_logits.cpu().detach().numpy() 
            legal_logits = np_logits.copy()
            np_move = move.cpu().detach().numpy()

            for i in range(len(position)): #iterate over the batch
                board = decode_board(position[i])
                mask = legal_moves(board)

                if sample_index % 1000 == 0 :
                    print(f"{sample_index} samples teated for legal move filtering")
                
                legal_logits[i] = mask*np_logits[i] #legal output

                sample_index+=1
                
                output.append(np_logits)
                legal_output.append(legal_logits)
                y_true.append(np_move)

    print("Computing for evaluation finished, concateanting arrays")
    output = np.concatenate(output, axis=0).squeeze()
    legal_output = np.concatenate(legal_output, axis=0).squeeze()
    y_true = np.concatenate(y_true, axis=0).squeeze()

    if cfg.legal_matching:
        print("Extracting top3 moves")
        top3_moves = extract_topk(output, k=3, axis=1)
        top3_legal = extract_topk(legal_output, k=3, axis=1)

        legal_top3_match = topk_accuracy(top3_moves,top3_legal, k=3, axis=1, threshold=0.5)   #threshold doesn't work

    print("Computing accuracy")
    pred_move = np.argmax(legal_output, axis=1)
    acc = accuracy_score(y_true, pred_move)

    print("Accuracy of Model:")
    print(acc)

    if cfg.legal_matching:
        print("Legal move matching in top3 pred moves:")
        print(legal_top3_match)

if __name__ == '__main__':
    dataset = LichessGames()    
    dataset.train_test(0.8,0.2)
    # train_model(dataset, CNN_resblock()
    eval_model(dataset, CNN_resblock())