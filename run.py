import gc
import torch
from metrics import dice_coefficient
from torch.utils.tensorboard import SummaryWriter


class ModelRun:
    def __init__(self, model, device=torch.device('cpu')):
        self.model = model
        self.device = device

    def _model_run(self, data_loader, stage, criterion, optimizer, clear_memory=False):
        '''
        Function of run the model with data loader generator
        '''
        loss_cum = 0
        dice_total = 0
        for data in data_loader:
            inputs = data['image']['data']
            labels = data['segmentation']['data']
            
            outputs = self.model(inputs.to(self.device))
            loss = criterion(outputs, labels.type(torch.FloatTensor).to(self.device))
            loss_cum += loss.item()
            dice = dice_coefficient(outputs, labels)
            dice_total += dice
            
            if stage=='train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if clear_memory:
                del data
                gc.collect()
                torch.cuda.empty_cache()
            
        return loss_cum / len(data_loader), dice_total / len(data_loader)

    def model_train(self, train_dataloader, valid_dataloader, epochs=1, lr=3e-4, epoch_freq=10):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        criterion = torch.nn.BCELoss()

        last_dice = 0.
        writer = SummaryWriter(log_dir='./output/')

        for epoch in range(epochs): 
            
            for stage in ['train', 'eval']:
                # TRAIN
                if stage=='train':
                    self.model.train()
                    loss_train, dice_train = self._model_run(train_dataloader, stage, criterion, optimizer)
                # EVALUATION
                elif stage=='eval':
                    self.model.eval()
                    with torch.no_grad():
                        loss_valid, dice_valid = self._model_run(valid_dataloader, stage, criterion, optimizer)
                        
            if dice_valid > last_dice:
                last_dice = dice_valid
                torch.save(self.model.state_dict(), f'./output/best_model.pth')   
                
            writer.add_scalar('train_loss', loss_train, epoch)
            writer.add_scalar('valid_loss', loss_valid, epoch)
            writer.add_scalar('dice_coef_train', dice_train, epoch)
            writer.add_scalar('dice_coef_valid', dice_valid, epoch)

            if (epoch % epoch_freq == 0):   
                print(f"Epoch: {epoch}, "
                    f"train_loss: {loss_train:.3f}, "
                    f"dice_train: {dice_train:.3f}, "
                    f"valid_loss: {loss_valid:.3f}, "
                    f"dice_valid: {dice_valid:.3f}"
                    )
                
        writer.close() 
        print(f'Training is finished!')
    