from Dataset import *
import torch.nn as nn
import torchvision
import torchvision.transforms as trasforms
import sys
from torch.nn import functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class sentinel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(sentinel, self).__init__()
        self.linear1 = nn.Linear(input_dim,32)
        self.relu1 = nn.Hardswish()
        self.linear2 = nn.Linear(32,128)
        self.relu2 = nn.Tanhshrink()
        self.linear3 = nn.Linear(128,32)
        self.relu3 = nn.Hardswish()
        self.linear4 = nn.Linear(32,output_dim)

    def forward(self, multi):
        out = self.linear1(multi)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        return out

class dem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(dem, self).__init__()
        self.linear1 = nn.Linear(input_dim,32)
        self.relu1 = nn.Hardswish()
        self.linear2 = nn.Linear(32,128)
        self.relu2 = nn.Tanhshrink()
        self.linear3 = nn.Linear(128,32)
        self.relu3 = nn.Hardswish()
        self.linear4 = nn.Linear(32,output_dim)

    def forward(self, geomo):
        out = self.linear1(geomo)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        return out

class sentinel_DEM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(sentinel_DEM, self).__init__()
        self.linear1 = nn.Linear(input_dim,32)
        self.relu1 = nn.Hardswish()
        self.linear2 = nn.Linear(32,128)
        self.relu2 = nn.Tanhshrink()
        self.linear3 = nn.Linear(128,32)
        self.relu3 = nn.Hardswish()
        self.linear4 = nn.Linear(32,output_dim)


    def forward(self, multi, geomo):
        x = torch.cat((multi, geomo), 1)
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        return out

def train_val_model(train, valid, model, criterion, optimizer, path_to_save, DEM=False, epochs=400):

    best_val = None

    for epoch in range(epochs):
        train_loss = []
        for i, (hyper, multi, chemi, geomo) in enumerate(train):
            multi = multi.reshape((batch_size, 21))
            hyper = hyper.reshape((batch_size, 4200))
            optimizer.zero_grad()
            if DEM:
                output = model(multi, geomo)
            else:
                output = model(multi)
            loss = criterion(output, chemi)
            loss.backward()
            optimizer.step()
            train_loss.append(loss)

        avg_loss = sum(train_loss)/len(train_loss)

        model.eval()
        val_loss = []
        for i, (hyper, multi, chemi, geomo) in enumerate(valid):
            multi = multi.reshape((batch_size, 21))
            hyper = hyper.reshape((batch_size, 4200))
            with torch.no_grad():
                if DEM:
                    output = model(multi, geomo)
                else:
                    output = model(multi)
                loss = criterion(output, chemi)
                val_loss.append(loss)

        avg_val_loss = sum(val_loss)/len(val_loss)

        print(f"Epoch: {epoch + 1}, train loss: {avg_loss:.3f}, validation loss: {avg_val_loss:.3f}")

        if epoch == 0 or avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                "net" :  model.state_dict(),
                "optim" : optimizer.state_dict(),
                "epoch" : epoch
            },
            path_to_save)

        model.train()

def test_model(test, model, criterion, path_model, DEM=False):
    state = torch.load(path_model)
    model.load_state_dict(state['net'])

    test_loss_all = []
    output_all = []
    chemi_all = []
    for i, (hyper, multi, chemi, geomo) in enumerate(test):

        multi = multi.reshape((batch_size, 21))
        hyper = hyper.reshape((batch_size, 4200))
        with torch.no_grad():
            if DEM:
                output = model(multi, geomo)
            else:
                output = model(multi)
            loss = criterion(output, chemi)
            test_loss_all.append(loss)

            output_all.append(output)
            chemi_all.append(chemi)

    y_pred = torch.cat(output_all, 0)
    y_pred = y_pred.squeeze()
    y_true = torch.cat(chemi_all, 0)


    return y_true.numpy(), y_pred.numpy()

def performances(y_true, y_pred, path_results):
    mae_all = []
    rmse_all = []
    r2_all = []
    for i in range(12):
        MAE = mean_absolute_error(y_true[:,i], y_pred[:,i])
        RMSE = mean_squared_error(y_true[:,i], y_pred[:,i])
        R2 = r2_score(y_true[:,i], y_pred[:,i])
        mae_all.append(MAE)
        rmse_all.append(RMSE**.5)
        r2_all.append(R2)
    # print(f"Average test loss: {sum(test_loss_all)/len(test_loss_all):.2f}")
    char = ['coarse','clay','silt','sand','pH.in.CaCl','pH.in.H2O','OC','CaCO3','N','P','K','CEC']
    final = pd.DataFrame({"Char": char,
     "RMSE": rmse_all, "MAE" : mae_all, "R2" : r2_all})
    final.to_csv(path_results)


csv_train = 'Data/Base_train.csv'
csv_val = 'Data/Base_val.csv'
csv_test = 'Data/Base_test.csv'

hyper_norm = InstanceStandardization()
multi_norm = InstanceStandardization()
chemicals_norm = VariableStandardization(12)
geomorpho_norm = VariableStandardization(8)

batch_size = 100

data = Dataset(csv_train, batch_size=batch_size, hyper_norm = hyper_norm, multi_norm = multi_norm, chemicals_norm = chemicals_norm, geomorpho_norm = geomorpho_norm)
print("Train read!")
data_val = Dataset(csv_val, batch_size=batch_size, hyper_norm = hyper_norm, multi_norm = multi_norm, chemicals_norm = chemicals_norm, geomorpho_norm = geomorpho_norm)
print("Validation read!")
data_test = Dataset(csv_test, batch_size=batch_size, hyper_norm = hyper_norm, multi_norm = multi_norm, chemicals_norm = chemicals_norm, geomorpho_norm = geomorpho_norm)
print("Test read!")


input_feat = data.multi_vals.shape[1]
output_feat = data.chemicals.shape[1]

model = sentinel(input_feat, output_feat)

lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# train_val_model(data,
#                 data_val,
#                 model,
#                 criterion,
#                 optimizer,
#                 path_to_save="models_save/nn_models/best_model_multi_target.pth",
#                 DEM=False)

# y_true, y_pred = test_model(data_test,
#                             model,
#                             criterion,
#                             path_model='models_save/nn_models/best_model_multi_target.pth',
#                             DEM=False)
#
# performances(y_true, y_pred, path_results="results/nn_models/NN_multitarget.csv")

input_feat = data.multi_vals.shape[1] + data.geomorpho.shape[1]
output_feat = data.chemicals.shape[1]

model = sentinel_DEM(input_feat, output_feat)

train_val_model(data,
                data_val,
                model,
                criterion,
                optimizer,
                path_to_save="models_save/nn_models/best_model_multi_target_DEM.pth",
                DEM=True)

y_true, y_pred = test_model(data_test,
                            model,
                            criterion,
                            path_model='models_save/nn_models/best_model_multi_target_DEM.pth',
                            DEM=True)

# performances(y_true, y_pred, path_results="results/nn_models/NN_multitarget_DEM.csv")
performances(y_true, y_pred, path_results="testing.csv")
