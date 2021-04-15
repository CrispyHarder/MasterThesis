import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
import argparse

save_path = os.path.join('storage','runs','test')

writer = SummaryWriter(log_dir=save_path)

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar('Const/train',5,epoch)
        writer.add_scalar('const/test',10,epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush()