
import torch
from torch import nn

class TinyVGG(nn.Module):
  """Cria a arquitetura TinyVGG.

  Imita a arquitetura TinyVGG utilizada no website CNN explainer em PyTorch.
  https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: número de canais de entrada
    hidden_units: número de kernels de convolução nas camadas conv2d
    output_shape: número de classes/saídas
  """

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, # how big is the square that's going over the image?
                    stride=1, # default
                    padding="same"), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2) # default stride value is same as kernel_size
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=2048, #descoberto com o truque de gerar uma entrada fictícia
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- acelera a execução graças a operator fusion


class MiniResNet(nn.Module):
  """Cria uma versão bem enxuta da arquitetura ResNet.

  Imita a arquitetura ResNet-18, mas reduz bastante a quantidade de camadas/blocos. 
    
  Args:
    input_shape: número de canais de entrada
    base_kernel_units: número de kernels de convolução do primeiro bloco da ResNet --> seguindo o padrão da arquitetura, a quantidade é dobrada após um conjunto (convolução + identidade). 
    output_shape: número de classes/saídas
  """

  def __init__(self, input_shape: int, base_kernel_units: int, output_shape: int) -> None:
      super().__init__()

      """ Bloco inicial: conv2d + BatchNorm + ReLU + MaxPool """
      self.input_block = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,out_channels=base_kernel_units,kernel_size=(5,5),stride=2,padding=2),
          nn.BatchNorm2d(num_features=base_kernel_units),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=1)
      )

      """ Primeira parte do bloco 1"""
      self.conv_block_1_1 = nn.Sequential(
          nn.Conv2d(in_channels=base_kernel_units,out_channels=base_kernel_units,kernel_size=(3,3),stride=1,padding=1),
          nn.BatchNorm2d(num_features=base_kernel_units),
          nn.ReLU(),
      )

      """ Segunda parte do bloco 1 """
      self.conv_block_1_2 = nn.Sequential(
          nn.Conv2d(in_channels=base_kernel_units,out_channels=base_kernel_units,kernel_size=(3,3),stride=1,padding=1),
          nn.BatchNorm2d(num_features=base_kernel_units),
      )

      """ Primeira parte do bloco 2: dobramos a quantidade de canais/kernels, mas reduzimos as dimensões por 2"""
      self.conv_block_2_1 = nn.Sequential(
          nn.Conv2d(in_channels=base_kernel_units,out_channels=2*base_kernel_units,kernel_size=(3,3),stride=2,padding=1),
          nn.BatchNorm2d(num_features=2*base_kernel_units),
          nn.ReLU(),
      )

      """ Segunda parte do bloco 2 """
      self.conv_block_2_2 = nn.Sequential(
          nn.Conv2d(in_channels=2*base_kernel_units,out_channels=2*base_kernel_units,kernel_size=(3,3),stride=1,padding=1),
          nn.BatchNorm2d(num_features=2*base_kernel_units),
      )

      """ Camada de convolução para a skip-connection do bloco 2 """
      self.concat_adjust = nn.Conv2d(in_channels=base_kernel_units,out_channels=2*base_kernel_units,kernel_size=(1,1),stride=(2,2),padding=0)
    
      self.ReLU = nn.ReLU()

      """ Global Average Pooling """
      self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))

      self.classifier = nn.Sequential(
          nn.Linear(in_features=2*base_kernel_units,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):

      #bloco inicial
      in_block_1 = self.input_block(x)

      #entrada do bloco 2 = saída do bloco 1: ReLU(forward_path + skip connection)   
      in_block_2 = self.ReLU(self.conv_block_1_2(self.conv_block_1_1(in_block_1)) + in_block_1)

      #saída do bloco 2 = ReLU(forward_path + skip_connection_com_conv2d)
      out_block_2 = self.ReLU(self.conv_block_2_2(self.conv_block_2_1(in_block_2)) + self.concat_adjust(in_block_2))
      
      #entrada do bloco classifier = resultado do Global Avg. Pool - reduz as dimensões espaciais para (1,1) --> REQUER squeeze() depois para removê-las
      in_class = self.GlobalAvgPool(out_block_2)
      return self.classifier(in_class.squeeze(dim=(2,3)))

