import logging
import torch
from torch import nn
from torch.nn import functional as F
from client.base.baseTrainer import BaseTrainer
from model.FCNN import createFCNNModel  # Make sure this is the correct import

logger = logging.getLogger(__name__)

class SignTrainer(BaseTrainer):
    def __init__(self, model, dataloader, criterion, args={}, watermarks={}):
        super().__init__(model, dataloader, criterion, args, watermarks)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def _train_epoch(self, epoch):
        model = self.model
        args = self.args
        device = args.get("device", 'cuda')  # Default to 'cuda' if not specified

        model.to(device)
        model.train()

        batch_loss = []
        batch_sign_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)
            sign_loss = torch.tensor(0.).to(device)
            if self.watermarks:
                sign_loss += SignLoss(self.watermarks, model).get_loss()  # Assuming 'self.watermarks' structure is compatible
            (loss + sign_loss).backward()

            # Uncomment to avoid NaN loss
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()
            batch_loss.append(loss.item())
            batch_sign_loss.append(sign_loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0.0
        epoch_sign_loss = sum(batch_sign_loss) / len(batch_sign_loss) if batch_sign_loss else 0.0
        return {'loss': epoch_loss, 'sign_loss': epoch_sign_loss}

class SignLoss():
    def __init__(self, watermarks, model):
        self.alpha = 0.2  # Loss scaling factor
        self.model = model
        self.watermarks = watermarks
        self.loss = 0

    def get_loss(self):
        self.loss = 0
        for key, wm in self.watermarks.items():
            if wm['flag']:
                b = wm['b'].to(torch.device('cuda'))
                M = wm['M'].to(torch.device('cuda'))
                # Modify below to suit model's specific layers if applicable
                target_feature = getattr(self.model, key)
                if hasattr(target_feature, 'weight'):
                    layer_output = target_feature.weight.mm(M)
                    if 'scheme' in wm and wm['scheme'] == 'BCE':
                        target = b.float()  # Assuming b is binary {0, 1}
                        self.loss += self.alpha * F.binary_cross_entropy_with_logits(layer_output.view(-1), target)
                    else:
                        self.loss += self.alpha * F.relu(-layer_output.mul(b.view(-1))).sum()

        return self.loss

# Example of how to initialize and use SignTrainer
# model = createFCNNModel()
# dataloader = ...  # DataLoader setup
# trainer = SignTrainer(model, dataloader, torch.nn.CrossEntropyLoss(), args={'device': 'cuda'}, watermarks={})
