# Tutorial: Come allenare YOLOv# sul cluster di AImagelab

> **Declino ogni responsabilità per qualisasi danno causato seguendo questa guida :)**  
<br>
> **In questo tutorial farò riferimento a _yolov8m_ ma il procedimento sarebbe identico per qualsiasi altro modello di [yolo](https://docs.ultralytics.com/models/)**

---

## Steps Preliminari

### 1) Richiedere accesso al cluster  
Apri un ticket su:  
[https://ailb-web.ing.unimore.it/tickets/](https://ailb-web.ing.unimore.it/tickets/)

### 2) Setup VPN  
Per accedere al cluster da remoto, usa la tua VPN preferita. Io ho usato **openvpn**:

```bash
$ sudo openfortivpn -c /etc/openfortivpn/config
```

Il contenuto di `/etc/openfortivpn/config` deve essere:

```
### configuration file for openfortivpn, see man openfortivpn(1) ###
# host = vpn.example.org
# port = 443
# username = vpnuser
# password = VPNpassw0rd
host = vpn.unimore.it
port = 443
username = <il_tuo_numero_della_mail>
```

> Ti consiglio di **non** inserire il campo `password` nel file di config: ti verrà chiesto automaticamente quando proverai ad attivare la vpn.

### 3) Accesso al cluster

Una volta attivata la VPN, puoi accedere al cluster da qualsiasi parte del mondo (con un accesso a internet) con il comando:

```bash
$ ssh <your-aimagelab-username>@ailb-login-02.ing.unimore.it
```

---

## Steps Effettivi

### 4) Prepara il dataset

Congratulazioni, ora hai sei dentro una shell nel cluster di AImageLab.  
    Adesso puoi creare una cartella dove manterrai il tuo dataset:

```bash
$ cd /work/cvcs2025
$ mkdir <nome_dir>
$ cd <nome_dir>
$ mkdir dataset
```

Ora puoi trasferire il tuo dataset in [formato YOLO](https://docs.ultralytics.com/datasets/detect/) nella cartella `dataset`, apri un altro termiale e esegui il comando:

```bash
$ scp -r path/to/my/local_dataset <your-aimagelab-username>@ailb-login-02.ing.unimore.it:/work/cvcs2025/<nome_dir>/dataset
```

>Se il tuo dataset **non è in formato YOLO**, puoi usare [Roboflow](https://roboflow.com/formats) per convertirlo.

Ora assicurati che in `/work/cvcs2025/<nome_dir>/dataset` siano presenti i file:
- `data.yaml`
- `train/`
- `test/`
- `valid/`

---

### 5) Crea un ambiente virtuale

Ora che hai sistemato il tuo dataset non ti resta che allenare il tuo modello.  

A tale scopo devi prima creare un ambiente virtuale tramite Anaconda. 

Nella tua home sul nodo di login esegui:

```bash
$ conda create --name <nome_env> python=3.10
$ conda activate <nome_env>
$ pip install ultralytics
```

> Optional: installa tutte le altre librerie che ti servono

>  __ATTENZIONE: HAI UNA QUOTA DI MEMORIA LIMITATA__ che puoi controllare in qualsiasi momento con il comando `squota`.   
Quando installi della roba nel tuo virtual environment se ci installi troppa roba potresti finire la tua quota di memoria senza accorgernete.  
Nel caso ti comparirà un errore del tipo:

```bash
ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded
```

Non temere, è probabile che tu non abbia fatto nulla di irreparabile, semplicemente rimuovi il tuo environment
con:

```bash
$ conda deactivate
$ conda env remove --name <nome_env>
```

Verifica che la memoria non sia ancora sforata con `squota` e prova a fare un nuovo enviroment con meno roba dentro. 

> Optional: usa `ncdu` per visualizzare l’uso del disco: `ncdu ~`  

> Optional: pulisci la cache di anaconda

---

### 6) Training del modello YOLO

Ora che hai a disposizione un environment con tutto quello che ti serve puoi richiedere al cluster di eseguire un training:

- Crea le cartelle nella tua home:

    ```bash
    $ mkdir ~/yolo
    $ mkdir ~/yolo/yolo_outputs
    ```
- Crea il file `yolo_train.sh` nella cartella `yolo`:

    ```bash
    $ vim ~/yolo/yolo_train.sh
    ```

    Contenuto del file:

    ```bash
    #!/bin/bash
    #SBATCH --job-name=yolo_training_<username>
    #SBATCH --output=yolo_train.out
    #SBATCH --error=yolo_train.err
    #SBATCH --account=cvcs2025
    #SBATCH --partition=all_usr_prod
    #SBATCH --ntasks=1
    #SBATCH --time=24:00:00
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32G

    module load cuda/11.8
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cv

    DATA_PATH="/work/cvcs2025/<group_name>/datasets/<nome_dataset>/data.yaml"
    MODEL="yolov8m.pt"
    OUTPUT_DIR=/homes/<username>/yolo/yolo_outputs

    yolo detect train model=$MODEL data=$DATA_PATH epochs=200 imgsz=960 batch=8 device=0 workers=8 project=$OUTPUT_DIR save_period=25 scale=0.2 shear=4 hsv_h=0.05 hsv_s=0.05 hsv_v=0.15 mixup=0.10 copy_paste=0.10 patience=100 mosaic=0.5

    ```

    > Ti consiglio di capire cosa significa ogni riga in modo da porter personalizzare questo script in base alle tue esigenze, è molto probabile che tu debba modificare delle impostazioni.  

    > Optional: aggiungi delle opzioni extra al comando `yolo detect train ...` per salvare checkpoints ogni tot epochs (`save_period=<n_epochs>`) , settare early stopping (`patience=<n_epochs>`), dropout (`dropout=<tot>`), data aumentation (`hsv_*, close_mosaic, affine transformations, ...`) ecc...

- Ora che hai creato lo script puoi chiedere a SLURM di eseguirlo su un nodo computazionale:

    ```bash
    $ sbatch ~/yolo/yolo_train.sh
    ```

- Puoi monitorare lo stato del job  con il comando:

    ```bash
    $ watch squeue -u <il_tuo_username>
    ```
    e puoi vedere i messaggi di **errore** o **log** dentro i files _yolo_train.err_ e _yolo_train.out_ monitorandoli con il comando:
    ```bash
    $ tail -f yolo_train.out
    ```
    ```bash
    $ tail -f yolo_train.err
    ```

    > Potrebbe darti alcuni problemi del tipo che il nodo computazionale ha troppa poca RAM/VRAM o roba del genere, in quel caso devi capire tu cosa va modificato nello script in base a quello che ti dice.  

- Quando il comando `squeue -u <il_tuo_username>` restituisce **vuoto**, il training è finito.

- Ora vai nella directory di output:

    ```bash
    $ cd ~/yolo/yolo_outputs/train
    ```

    Qui dentro c'è un sacco di roba interessante tra cui anche la cartella `weights` dentro cui trovi:  

    `best.pt`: pesi del modello con le migliori performance  
    `last.pt`: ultimo set di pesi utilizzato  

- Nel caso in cui il training sia terminato prima di raggiungere il numero di epochs prestabilito, perchè ad esempio hai finito il tempo a tua disposizione, puoi riprenderlo utilizzando `sbatch` dello script `yolo_resume_training.sh`:

    ``` bash
    #!/bin/bash
    #SBATCH --job-name=yolo_training_<username>
    #SBATCH --output=yolo_train.out
    #SBATCH --error=yolo_train.err
    #SBATCH --account=cvcs2025
    #SBATCH --partition=all_usr_prod
    #SBATCH --ntasks=1
    #SBATCH --time=1:00:00
    #SBATCH --cpus-per-task=4
    #SBATCH --gres=gpu:1
    #SBATCH --mem=16G

    module load cuda/11.8
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cv

    DATA_PATH="/work/cvcs2025/<group_name>/datasets/<nome_dataset>/data.yaml"
    MODEL="yolov8m.pt"
    OUTPUT_DIR=/homes/<username>/yolo/yolo_outputs

    yolo detect train model=$OUTPUT_DIR/train<N>/weights/last.pt resume=True device=0
    ```
    
    Cerca di matchare le opzioni `#SBATCH ...` del file che hai usato per il training

    <br>  
    <br>

    <span style="background-color:#e0f7fa; color:#00695c; padding:10px 18px; border-radius:20px; font-size:16px; font-weight:500;">Spero che questa guida ti sia stata utile, ciao :)</span>




