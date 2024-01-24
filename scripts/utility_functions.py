
import torch
from typing import Dict,List

""" Função que computa a quantidade de amostras de cada classe 

    Args:
    label_tensor: torch.Tensor - tensor que contém os rótulos das amostras de um dataset
    class_names: List - lista que contém os nomes (strings) das classes do dataset

    Retorna:
    dict: dicionário contendo "nome_classe": #ocorrências 
"""


def check_class_balance(label_tensor: torch.Tensor,
                        class_names: List) -> Dict[str, List[float]]:
    
    #extrai a sequência de rótulos numéricos do dataset
    lbl = torch.unique(label_tensor)  

    #inicializa um dicionário vazio
    dict = {}  

    #conta o número de ocorrências de cada rótulo no label_tensor -- rótulos são inteiros de 0, 1, ..., n_classes - 1
    for i in range(len(lbl)):
        
        #conta as ocorrências do rótulo i = {0, 1, ...} no array de labels do dataset
        ci = torch.count_nonzero(label_tensor == i)

        #acrescenta ao dicionário o nome da i-ésima classe junto da contagem de amostras dessa classe no dataset
        dict[class_names[i]] = ci.item()

    return dict

from typing import Dict, List
import matplotlib.pyplot as plt

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    val_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

import torch

""" Função que computa os logits e os labels estimados dado um modelo treinado e um conjunto de amostras de entrada 
"""
def predict_model(model: torch.nn.Module,
                  X: torch.Tensor,
                  task: str,
                  device=None,
                  ):

    # Put model in eval mode
    model.eval() 
  
    # Turn on inference context manager
    with torch.inference_mode():
      
        # Envia os dados para o dispositivo alvo (device)
        X = X.to(device)
        
        #Forward pass
        pred_logits = model(X)
        
        if task == "binary":
            pred_labels = torch.round(torch.sigmoid(pred_logits))
        else:
            pred_labels = torch.argmax(torch.softmax(pred_logits,dim=1))
          
    return pred_logits,pred_labels


def predict_model_dataloader(model: torch.nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             task: str,
                             n_classes: int,
                             device=None,
                            ):

    # Put model in eval mode
    model.eval() 
    
    #creates an empty torch.Tensor
    if task == "binary-class":
        all_prediction_logits = torch.empty((0,1)).to(device)
    elif task == "multi-class":
        all_prediction_logits = torch.empty((0,n_classes)).to(device)

    #empty Tensor to gather the prediction labels over the entire dataset
    all_prediction_labels = torch.empty((0,1)).to(device)
    all_true_labels = torch.empty((0,1)).to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
         
            # Envia os dados para o dispositivo alvo (device) e garante que os labels são do tipo torch.float32
            X, y = X.to(device), y.to(device)
  
            # 1. Forward pass
            pred_logits = model(X)
            all_prediction_logits = torch.cat((all_prediction_logits, pred_logits), dim=0)

            if task == "binary-class":
                pred_labels = torch.round(torch.sigmoid(pred_logits))
            else:
                pred_labels = torch.argmax(pred_logits,dim=1).unsqueeze(dim=1)

            #unsqueeze() - necessário para que pred_labels seja (n_batch,1)
            all_prediction_labels = torch.cat((all_prediction_labels, pred_labels), dim=0)

            all_true_labels = torch.cat((all_true_labels, y), dim=0)
          
    all_true_labels = all_true_labels.type(torch.long)
    return all_prediction_logits,all_prediction_labels,all_true_labels
