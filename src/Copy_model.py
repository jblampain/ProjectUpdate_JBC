import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #224x224x3 -> 224x224x16 -> maxpool 112x112x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),  #112x112x16 -> 112x112x32 -> maxpool 56x56x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            
            nn.Conv2d(32, 64, 3, padding=1),  #56x56x32 -> 56x56x64 -> maxpool 28x28x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            # Since we are using BatchNorm and data augmentation,
            # we can go deeper than before and add one more conv layer
            nn.Conv2d(64, 128, 3, padding=1),  #28x28x64 -> 28x28x128 -> maxpool 14x14x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
            
        self.classifier = nn.Sequential(   
            nn.Flatten(),  
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature Extraction
        x = self.features(x)
        
        # Classification
        x = self.classifier(x)
        return x
         

# Instantiate the model
# num_classes = 10  # Change this based on the number of landmark classes
  model = MyModel(num_classes)

# Print the model architecture
  print(model)




        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
