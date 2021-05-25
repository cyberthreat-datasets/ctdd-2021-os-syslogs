import argparse
import Transformer as tnsf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

#     parser.add_argument('--log_normal', default='hdfs_test_normal', type=str, help='parsed log file of normal testing data')
#     parser.add_argument('--log_abnormal', default='hdfs_test_abnormal', type=str, help='parsed log file of abnormal testing data')
    
    parser.add_argument('--log_normal', default='linux_test_normal', type=str, help='parsed log file of normal testing data')
    parser.add_argument('--log_abnormal', default='linux_abnormal', type=str, help='parsed log file of abnormal testing data')
    
    parser.add_argument('--window_size', default=10, type=int, help='lenght of training window')
    parser.add_argument('--num_candidates', default=10, type=int, help='number of candidates considered correct predict')

    parser.add_argument('--federated', default=False, type=bool, help='number of gpus of gpus to train')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of gpus of gpus to train')
    parser.add_argument('--model_dir', default='../Saved_Models', type=str, help='the directory to store the model')
    parser.add_argument('--model_file', default='', type=str, help='the file of previously trained model')    
#     parser.add_argument('--data_dir', default='../HDFS_Dataset/', type=str, help='the directory where training data is stored')
    parser.add_argument('--data_dir', default='../CTDD_Dataset/Sample_Dataset_Train_Test_Log_Keys', type=str, help='the directory where training data is stored')
    
    args = parser.parse_args()

    tnsf.test(args)

