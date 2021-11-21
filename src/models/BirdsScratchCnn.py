"""Birds model from scratch."""

from torch import nn
from torchsummary import summary
from config import Config


BirdsInputSize = (3, Config.BIRDS_IMG_WIDTH, Config.BIRDS_IMG_HEIGHT)
BirdsScratchCnn = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Conv2d(10, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Conv2d(10, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Conv2d(10, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Conv2d(10, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Conv2d(10, 10, kernel_size=(3, 3), stride=1, padding="same", padding_mode="zeros"),
    nn.Flatten(),
    nn.Linear(501760, 7),
    nn.Softmax(-1),
)


print(summary(BirdsScratchCnn, input_size=BirdsInputSize, device="cpu"))
