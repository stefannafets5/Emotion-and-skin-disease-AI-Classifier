import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes, use_dropout=True):
        super(SimpleMLP, self).__init__()
        dropout_layer1 = nn.Identity()
        dropout_layer2 = nn.Identity()
        if use_dropout :
            dropout_layer1 = nn.Dropout(p=0.3)
            dropout_layer2 = nn.Dropout(p=0.3)
        self.net = nn.Sequential(
            nn.Flatten(),
            # Layer 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            dropout_layer1,
            
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            dropout_layer2,
            
            # Output Layer
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, use_batch_norm=True):
        super(SimpleCNN, self).__init__()

        bn1_layer = nn.Identity()
        bn2_layer = nn.Identity()
        bn3_layer = nn.Identity()
        bn4_layer = nn.Identity()

        if (use_batch_norm):
            bn1_layer = nn.BatchNorm2d(32)
            bn2_layer = nn.BatchNorm2d(64)
            bn3_layer = nn.BatchNorm2d(128)
            bn4_layer = nn.BatchNorm2d(128)
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn1 = bn1_layer
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn2 = bn2_layer
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn3 = bn3_layer
        
        # Block 4: Conv -> BN -> ReLU
        # (Using 4 conv layers as requested)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=not use_batch_norm)
        self.bn4 = bn4_layer
        
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global Pooling + Flatten
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing models on {device}...\n")
    
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    
    mlp = SimpleMLP(3*128*128, 7).to(device)
    cnn = SimpleCNN(7).to(device)
    
    print(f"MLP Output Shape: {mlp(dummy_input).shape}")
    print(f"CNN Output Shape: {cnn(dummy_input).shape}")
