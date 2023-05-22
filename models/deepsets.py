import torch.nn as nn
import torch
import ipdb

class DeepSets(nn.Module):
    def __init__(self, nclass=8, embed_size=128):
        super(DeepSets, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1, embed_size),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embed_size, nclass + 1)
        )
        

    def get_readout_layer(self):
        raise NotImplementedError()

    def get_embed_size(self):
        raise NotImplementedError()

    def forward(self, data, **kwargs):
        x = data['data']
        msk = data['mask']
        x = self.encoder(x.unsqueeze(-1))
        x = (x * msk.unsqueeze(-1)).sum(-2)
        return self.decoder(x)

    def np_pred(self, pred, candidate_set):
        with torch.no_grad():
            pred = torch.tensor(pred, dtype=torch.float)
            candidate_set = torch.tensor(candidate_set, dtype=torch.float)
            return self.forward({"data": pred, "mask": candidate_set}).numpy()


class TrainedThreshold(nn.Module):
    def __init__(self, nclass, embed_size=128):
        super().__init__()
        self.fc1 = nn.Linear(nclass, embed_size)
        self.fc3 = nn.Linear(embed_size, 2)
        self.act = torch.sigmoid

    def forward(self, data, **kwargs):
        x = self.act(self.fc1(data))
        return {'logit': data, 'out': self.fc3(x)}

if __name__ == '__main__':
    nclass = 5
    y = torch.rand(1,nclass)
    msk = torch.randint(0, 2, (1, nclass))
    model = DeepSets(nclass)
    out = model({'data': y, 'mask': msk})
