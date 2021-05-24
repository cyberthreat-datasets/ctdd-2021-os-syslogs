import argparse
import Transformer as tnsf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#     parser.add_argument('--log_file', default='hdfs_train', type=str, help='parsed log file')
#     parser.add_argument('--log_normal', default='HDFS/hdfs_test_normal', type=str, help='parsed log file of normal testing data')
#     parser.add_argument('--log_abnormal', default='HDFS/hdfs_test_abnormal', type=str, help='parsed log file of abnormal testing data')
    
    parser.add_argument('--log_file', default='linux_train', type=str, help='parsed log file')
    parser.add_argument('--log_normal', default='linux_test_normal', type=str, help='parsed log file of normal testing data')
    parser.add_argument('--log_abnormal', default='linux_abnormal', type=str, help='parsed log file of abnormal testing data')
    
    parser.add_argument('--clients', default=2, type=int, help='number of clients')
    parser.add_argument('--rounds', default=2, type=int, help='number of rounds')
    parser.add_argument('--frac', default=1.0, type=float, help='percentage of users to use per round')

    parser.add_argument('--batch_size', default=512, type=int, help='input batch size for training')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train')
    parser.add_argument('--window_size', default=10, type=int, help='lenght of training window')

    parser.add_argument('--dropout', default=0.2, type=float, help='number of epochs to train')    
    parser.add_argument('--num_layers', default=1, type=int, help='number of encoder and decoders')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--num_classes', type=int, help='number of total log keys')
    parser.add_argument('--num_candidates', default=10, type=int, help='number of predictors sequence as correct predict')

    parser.add_argument('--federated', default=True, type=bool, help='number of gpus of gpus to train')    
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus of gpus to train')
    parser.add_argument('--model_dir', default='Model', type=str, help='the directory to store the model')
    parser.add_argument('--data_dir', default='Dataset/Linux', type=str, help='the directory where training data is stored')

    args = parser.parse_args()

    tnsf.federated_training(args)
