# Cyber Threat Detection Dataset

A Pytorch implementation of Interpretable Federated Transformer Log Learning for Threat Forensics.

Dataset

The original HDFS logs can be found http://people.iiis.tsinghua.edu.cn/~weixu/sospdata.html.

Running the experiments

The baseline experiment trains the model in the conventional way.

To run the baseline experiment :

   python train.py --num_classes=449 --num_layers=4 --num_heads=2 --epochs=5 --batch_size=2048

Federated experiment involves training a global model using many local models.

To run the federated experiment :

   python federated_train.py --num_classes=449 --epochs=5 --batch_size=2048 --num_layers=4 --num_heads=2 --clients=4 --rounds=10

The required parameter for training is --num_classes, for HDFS it is 29 and for CTDD it is 449.

The default values for various paramters parsed to the experiment are given in train.py. Details are given some of those parameters:
   --log_file', default='Linux/linux_train', type=str, help='parsed log file'
   --log_normal', default='Linux/linux_test_normal', type=str, help='parsed log file of normal testing data'
   --log_abnormal', default='Linux/linux_abnormal', type=str, help='parsed log file of abnormal testing data'
    
   --window_size', default=10, type=int, help='lenght of training window'

   --batch_size', default=512, type=int, help='input batch size for training'
   --epochs', default=10, type=int, help='number of epochs to train'
    
   --dropout', default=0.2, type=float, help='number of epochs to train'
   --num_layers', default=1, type=int, help='number of encoder and decoders'
   --num_heads', default=1, type=int, help='number of heads'
   --seed', default=1, type=int, help='random seed'

   --num_classes', type=int, help='number of total log keys'
   --num_candidates', default=10, type=int, help='number of predictors sequence as correct predict'
    
   --federated', default=False, type=bool, help='federated involved'      
   --num_gpus', default=1, type=int, help='number of gpus of gpus to train'
   --model_dir', default='Model', type=str, help='the directory to store the model'
   --data_dir', default='Dataset', type=str, help='the directory where training data is stored'

Additional Federated Parameters:
    
   --clients', default=2, type=int, help='number of clients'
   --rounds', default=2, type=int, help='number of rounds'
   --frac', default=1.0, type=float, help='percentage of users to use per round'

The notebook detection.ipynb is for easier experimentation along with exposure to:
    data, log templates, log sequences, model training and evaluation

The notebook parser.ipynb is to be used for:
    generating data, log key sequences, given log files
    splitting dataset into n number of clients for federated experiments
    changing length of sequence by time with time_seq 
    
The notebook pkl_to_logfile.ipynb is for extracting logs in pkl to csv.
    