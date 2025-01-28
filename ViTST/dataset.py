import  yfinance as yf

import torch
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def setup_logger(log_level=logging.DEBUG):
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[
            logging.StreamHandler(), 
        ]
    )

def load_dataset(stocks='005930.KS', start='2021-01-01', end='2021-05-31'): 
    df = yf.download(stocks, start=start, end=end)

    X = df.drop('Close', axis=1)
    y = df[['Close']]

    ms = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(X)
    y_ms = ms.fit_transform(y)

    X_train = X_ss[:79, :]
    X_test = X_ss[79:, :]

    y_train = y_ms[:79, :]
    y_test = y_ms[79:, :]

    X_train_tensors = torch.Tensor(X_train)
    X_test_tensors = torch.Tensor(X_test)

    y_train_tensors = torch.Tensor(y_train)
    y_test_tensors = torch.Tensor(y_test)

    X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    setup_logger(log_level=logging.INFO)

    logging.info(f"X_train_tensors_f Shape: {X_train_tensors_f.shape}")
    logging.info(f"X_test_tensors_f Shape: {X_test_tensors_f.shape}")

    return X_train_tensors_f, y_train_tensors

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

def timeseries_to_images(X, y, time_window=20, step=10, img_size=(224, 224)):
    images = []

    if isinstance(X, torch.Tensor):
        X = X.squeeze(1)  
        X = pd.DataFrame(X.numpy())  
    if isinstance(y, torch.Tensor):
        y = pd.DataFrame(y.numpy(), columns=['Close'])

    time_index = list(range(len(X)))  

    colors = ["blue", "green", "purple", "orange"]

    for start in range(0, len(X) - time_window, step):
        end = start + time_window

        for i, col in enumerate(X.columns):
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.set_facecolor('white')  
            
            ax.plot(time_index[start:end], X[col].iloc[start:end], color=colors[i % len(colors)], label=col, linewidth=2)

            ax.grid(True, linestyle="--", linewidth=0.5)  
            ax.set_xticks(time_index[start:end:5])  
            ax.set_xticklabels(time_index[start:end:5], rotation=45, fontsize=6) 

            y_min = X[col].iloc[start:end].min()
            y_max = X[col].iloc[start:end].max()
            yticks = np.linspace(y_min, y_max, num=5)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{val:.2f}" for val in yticks], fontsize=6) 

            # ax.legend(loc="upper right", fontsize=6)  
            ax.axis("on")

            fig.canvas.draw()
            img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            img = img.resize(img_size)
            img = transforms.ToTensor()(img)

            if img.shape[0] == 1:  
                img = img.repeat(3, 1, 1)

            images.append(img)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_facecolor('white')  
        ax.plot(time_index[start:end], y.iloc[start:end], color="black", label="Close Price", linewidth=2)

        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_xticks(time_index[start:end:5])
        ax.set_xticklabels(time_index[start:end:5], rotation=45, fontsize=6)

        y_min = y.iloc[start:end].min().values.flatten()[0]  
        y_max = y.iloc[start:end].max().values.flatten()[0]  
        yticks = np.linspace(y_min, y_max, num=5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{val:.2f}" for val in yticks], fontsize=6)

        # ax.legend(loc="upper right", fontsize=6)
        ax.axis("on")

        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img = img.resize(img_size)
        img = transforms.ToTensor()(img)

        if img.shape[0] == 1:  
            img = img.repeat(3, 1, 1)

        images.append(img)
        plt.close(fig)

    return torch.stack(images) 

if __name__ == "__main__":
    X, y = load_dataset('005930.KS', '2021-01-01', '2021-05-31')

    vit_input = timeseries_to_images(X, y, time_window=20, step=10)  # to_pil_image(vit_input[0]).save("test_image.png") 
