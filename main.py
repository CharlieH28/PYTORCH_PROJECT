import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer

# torch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

### Load data into datasets

datalist = []
data_name_list = ["Alternative","Blues","Childrens Music","Comedy","Electronic","Folk","Hip-Hop","Movie","Ska","Soul"]
data_name_list_chart = ["Alt","Blues","Child","Com","Ele","Folk","Hip","Mov","Ska","Soul"]
music_alt_data = pd.read_csv("data/musicGENRE/alternative.csv")
music_blu_data = pd.read_csv("data/musicGENRE/blues.csv")
music_chd_data = pd.read_csv("data/musicGENRE/childrens music.csv")
music_com_data = pd.read_csv("data/musicGENRE/comedy.csv")
music_ele_data = pd.read_csv("data/musicGENRE/electronic.csv")
music_flk_data = pd.read_csv("data/musicGENRE/folk.csv")
music_hip_data = pd.read_csv("data/musicGENRE/hip-hop.csv")
music_mov_data = pd.read_csv("data/musicGENRE/movie.csv")
music_ska_data = pd.read_csv("data/musicGENRE/ska.csv")
music_sou_data = pd.read_csv("data/musicGENRE/soul.csv")

datalist.append(music_alt_data)
datalist.append(music_blu_data)
datalist.append(music_chd_data)
datalist.append(music_com_data)
datalist.append(music_ele_data)
datalist.append(music_flk_data)
datalist.append(music_hip_data)
datalist.append(music_mov_data)
datalist.append(music_ska_data)
datalist.append(music_sou_data)

test_data = pd.read_csv("data/musicGENRE/test.csv")

### Process Data -> combine data to a single dataframe
# step 1 combine data -> to one big dataframe && step 2 sort missing values

processed_d_list = []
for d in datalist:
    p_df = d.copy(deep=True)
    p_df.loc[p_df["tempo"] == "?", "tempo"] = None
    knn_imputer_temp = KNNImputer(n_neighbors=2, weights="uniform")
    p_df['tempo'] = knn_imputer_temp.fit_transform(p_df[['tempo']])
    
    p_df.loc[p_df["duration_ms"] == -1, "duration_ms"] = None
    knn_imputer_ms = KNNImputer(n_neighbors=2, weights="uniform")
    p_df['duration_ms'] = knn_imputer_ms.fit_transform(p_df[['duration_ms']])

    processed_d_list.append(p_df)
    
processed_df = pd.concat(processed_d_list)

without_id_names_df = processed_df.copy(deep=True)
labels = without_id_names_df['genre']
without_id_names_df = without_id_names_df.drop(columns=['artist_name', 'track_name','track_id','genre'])

processed_data_cat = without_id_names_df.select_dtypes(include=object)
processed_data_num = without_id_names_df.select_dtypes(exclude=object)

# step 3 sort out categorical data

processed_data_cat = pd.get_dummies(processed_data_cat)
frames = [processed_data_cat.reset_index(drop=True), processed_data_num.reset_index(drop=True),]

# x data
result = pd.concat(frames,axis=1)
nx= result.iloc[:,:-1].values 

# labels
ny=labels.tolist()
ly = LabelEncoder()
ny = ly.fit_transform(ny)

# normalize data (without class)
sc = StandardScaler()
X_scaled = sc.fit_transform(nx)


# split data
nx_train,nx_test,ny_train,ny_test = train_test_split(X_scaled,ny,test_size=0.3)

## 

### Neural network

# convert to tensor
x_train_to_tensor = torch.from_numpy(nx_train).to(torch.float32)
y_train_to_tensor = torch.from_numpy(ny_train).to(torch.long)
x_test_to_tensor = torch.from_numpy(nx_test).to(torch.float32)
y_test_to_tensor = torch.from_numpy(ny_test).to(torch.long)

# create TensorDataset for dataLoader
train_dataset = TensorDataset(x_train_to_tensor,y_train_to_tensor)
train_dataset = TensorDataset(x_test_to_tensor,y_test_to_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=16)
test_dataloader = DataLoader(train_dataset, batch_size=16)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Basic NN
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
# More advnaced NN -> explores more complex neural network weights.
class MultiClassNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_prob=0.2):
        super(MultiClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x



input_size = 29
hidden_size = 64  # Adjust as needed
hidden_size1 = 64  # Adjust as needed
hidden_size2 = 32  # Adjust as needed
output_size = 12 
#model = SimpleNN(input_size, hidden_size, output_size).to(device)
model = MultiClassNN(input_size, hidden_size1, hidden_size2, output_size).to(device)


### Train processed data

loss_fn = nn.CrossEntropyLoss()
criterion = nn.MSELoss()  # Mean Squared Error for regression, change for classification
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        #print(pred," y: ",y)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

### Test Model

