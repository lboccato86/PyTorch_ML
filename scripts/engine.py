
import torch
import torchmetrics
from typing import Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               task: str,
               metric: torchmetrics.Metric = None) -> Tuple[float, float]:
  
  """Treina um modelo PyTorch por uma única época

  Coloca o modelo PyTorch em modo de treinamento e, então, percorre todos os passos
  necessários para seu treinamento: forward pass, loss calculation, loss backward
  e optimizer step. 
  
  Args:
    model: um modelo PyTorch.
    dataloader: uma instância de Dataloader com a qual o modelo será treinado. 
    loss_fn: uma função custo PyTorch a ser minimizada.
    optimizer: um otimizador PyTorch para realizar a minimização da função custo. 
    device: dispositivo alvo no qual a computação ocorrerá (e.g. "cuda" or "cpu").
    task: string que indica o tipo de tarefa (e.g., "multi-class" ou "binary-class")
    metric: uma métrica PyTorch/torchmetrics para avaliação do modelo durante o treinamento.

  Retorna:
    Uma tupla contendo os valores da função custo de treinamento e da métrica de desempenho. 
    Padrão: (train_loss, train_accuracy). Exemplo:
    
    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()
  
  # Setup train loss and train metric values
  train_loss, train_metric = 0, 0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader): #batch = índice do batch atual recuperado; (X,y) = tensores com as amostras de treinamento e os rótulos ou as saídas desejadas
      
      # Envia os dados para o dispositivo alvo (device) e garante que os labels são do tipo torch.float32
      X, y = X.to(device), y.type(torch.float32).to(device)
    
      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      if task == 'multi-class' or task == 'multiclass': 
        
        """ Para a nn.CrossEntropyLoss(), os targets devem ser Long e não podem ter uma dimensão extra 
            Exemplo: batch_size = 32 --> (32,1) é o que vem da MedMNIST, em int32; precisamos converter em torch.long e (32) """ 
        y = y.squeeze().long()
            
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate performance metric across all batches
      if metric is not None:
        train_metric += metric(preds=y_pred,target=y).item()

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_metric = train_metric / len(dataloader)
  return train_loss, train_metric

#import torch
#import torchmetrics
#from typing import Tuple

def validation_step(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: torch.nn.Module,
                    device: torch.device,
                    task: str,
                    metric: torchmetrics.Metric = None) -> Tuple[float, float]:
  """ Avalia um modelo PyTorch em uma época sobre o conjunto de dados definido por um DataLoader.

  Coloca o modelo alvo PyTorch no modo "eval" e, então, realiza um
  forward pass no dataset. 

  Args:
    model: um modelo PyTorch a ser avaliado. 
    dataloader: uma instância DataLoader sobre a qual o modelo será avaliado. 
    loss_fn: uma função custo PyTorch para calcular a perda no conjunto (tipicamente de validação). 
    device: dispositivo alvo no qual a computação ocorrerá (e.g. "cuda" or "cpu").
    metric: uma métrica PyTorch/torchmetrics para avaliação do modelo.

  Retorna:
    Uma tupla contendo os valores da função custo e da métrica de desempenho no conjunto de dados
    fornecido (tipicamente de validação). 
    A tuple of validation loss and accuracy metrics.
    Padrão: (train_loss, train_accuracy). Exemplo:
     
    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 
  
  # Setup test loss and test accuracy values
  val_loss, val_metric = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          
          # Envia os dados para o dispositivo alvo (device) e garante que os labels são do tipo torch.float32
          X, y = X.to(device), y.type(torch.float32).to(device)
  
          # 1. Forward pass
          val_pred_logits = model(X)

          # 2. Calculate  and accumulate loss
          if task == 'multi-class' or task == 'multiclass':  
        
            """ Para a nn.CrossEntropyLoss(), os targets devem ser Long e não podem ter uma dimensão extra 
            Exemplo: batch_size = 32 --> (32,1) é o que vem da MedMNIST, em int32; precisamos converter em torch.long e (32) """ 
            y = y.squeeze().long()

          loss = loss_fn(val_pred_logits, y)
          val_loss += loss.item()
          
          # Calculate and accumulate metric
          if metric is not None:
            val_metric += metric(preds=val_pred_logits,target=y).item()
          
  # Adjust metrics to get average loss and accuracy per batch 
  val_loss = val_loss / len(dataloader)
  val_metric = val_metric / len(dataloader)
  return val_loss, val_metric

import torch
from pathlib import Path
from typing import Dict, List
from tqdm.auto import tqdm

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          patience: int,
          model_save_path: Path,
          task: str,
          device: torch.device,
          metric: torchmetrics.Metric = None) -> Dict[str, List[float]]:
  """Treina um modelo PyTorch enquanto monitora seu desempenho em um conjunto
  de validação. 

  Passa um modelo PyTorch através das funções train_step() e val_step() por um 
  número máximo de épocas, treinando (i.e., ajustando os seus parâmetros) e validando
  o modelo no mesmo loop. 

  Calcula, imprime e armazena métricas de avaliação durante o processo de treinamento.

  Args:
    model: um modelo PyTorch a ser treinado e validado. 
    train_dataloader: uma instância DataLoader com os dados de treinamento. 
    val_dataloader: uma instância DataLoader para avaliação da generalização do modelo PyTorch.
    optimizer: um otimizador PyTorch para realizar a minimização da função custo. 
    loss_fn: uma função custo PyTorch a ser minimizada.
    epochs: valor inteiro que indica por quantas épocas (no máximo) o modelo será treinado.
    patience: valor inteiro que indica por quantas épocas aceitamos que o modelo continue seu treinamento mesmo se a métrica/loss de validação estiver piorando. 
    model_save_path: caminho completo para o arquivo que guardará a melhor versão (pesos) do modelo treinado.
    device: dispositivo alvo no qual a computação ocorrerá (e.g. "cuda" or "cpu").
    metric: uma métrica PyTorch/torchmetrics para avaliação do modelo.
  
  Retorna:
    Um dicionário contendo os valores da função custo para os dados de treinamento e validação, além
    (opcionalmente) de uma métrica de desempenho adicional. Cada métrica possui um valor em uma lista
    referente a cada época. 
    Estrutura:   {train_loss: [...],
                  train_acc: [...],
                  val_loss: [...],
                  val_acc: [...]} 
    Por exemplo, se epochs = 2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  val_loss: [1.2641, 1.5706],
                  val_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "val_loss": [],
             "val_acc": []
  }
  
  # Initialize the best validation accuracy found (as 0) and the best epoch
  best_epoch = -1
  best_val_acc = -1

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          task=task,
                                          metric=metric)
      val_loss, val_acc = validation_step(model=model,
                                    dataloader=val_dataloader,
                                    loss_fn=loss_fn,
                                    device=device,
                                    task=task,
                                    metric=metric)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)

      # Early stopping procedure
      if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(obj=model.state_dict(),
                   f=model_save_path)
      elif (epoch - best_epoch) > patience:
         print(f"Early stopping: Interrupting training at epoch {epoch}")
         model.load_state_dict(torch.load(f=model_save_path))
         print(f"Reloading model parameters from epoch {best_epoch} - Max. Validation Metric = {best_val_acc:.4f}")
         break  # terminate the training loop

  # Return the filled results at the end of the epochs
  return results
