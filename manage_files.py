import os
import pandas as pd



def save_metrics(data, filename: str, save_directory: str = None):
  # Crée le chemin complet
  filepath = os.path.join(save_directory, filename)
    
  # Convertit tous les objets NumPy dans le dictionnaire
  data = pd.DataFrame(list(data.items()), columns=["Métrique", "Valeur"])
  data.to_csv(filepath)
    
  print(f'Metrics saved to {filepath}')
