import subprocess
import os
from tqdm import tqdm
import time

def run_reconContrast():
    sysmat_cmd = 'python3 reconContrast.py' 
    os.system(sysmat_cmd)
    for _ in tqdm(range(10), desc='Constructing Contrast'):
        time.sleep(0.5)
    


def plotCNR():
    reconstruction_cmd = 'python3 plotCNR.py'
    os.system(reconstruction_cmd)
    for _ in tqdm(range(10), desc='plotting CNR'):
        time.sleep(0.5)


def main():
 

    print('Constructing Contrast...')
    run_reconContrast()
    print('Contrast created.')


    print('Plotting CNR...')
    plotCNR()
    print('CNR Plotted')


if __name__ == '__main__':
    main()
