from math import log
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import importlib
import re
import pickle
from Parsers import Spell

importlib.reload(Spell)


def parse(log_source, log_file, algorithm):
    """
    Parses log file.

    Args:
        log_source: The source of the logs (e.g. HDFS, Openstack, Linux).
        log_file: The name of the log file.
        algorithm: Parsing algorithm: Spell or Drain.
    """
    
    input_dir = 'Dataset/' + log_source + "/"
    output_dir = algorithm + "_results/"
    
    st = 0.5
    
    #Parsing parameters
    # HDFS Logs
    if log_source == 'HDFS':
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
#         regex = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        regex      = [
            r'blk_(|-)[0-9]+' , # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]
        tau = 0.66
    #Linux Logs
    elif log_source == 'Linux':
        log_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
#         regex = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
        regex      = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}',
            r'\w{8}\-\w{4}\-\w{4}\-\w{4}\-\w{12}' # Kernel
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]   
        tau = 0.50
    #Openstack Logs
    elif log_source == 'Openstack':
#         log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] \[<Instance>\] <Content>'
        log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
#         regex = [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+', r'\w{8}-\w{4}-\w{4}-\w{4}-\w{12}']
        regex      = [
            r'((\d+\.){3}\d+,?)+', 
            r'/.+?\s', r'\d+', 
#             's/\b[a-z0-9]\{8\}-[a-z0-9]\{4\}-[a-z0-9]\{4\}-[a-z0-9]\{4\}-[a-z0-9]\{12\}\b/MASKED_ID/g'
            r'\w{8}\-\w{4}\-\w{4}\-\w{4}\-\w{12}'
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]  
        tau = 0.66

    #Initialize parser
    parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parsed_logs = parser.parse(log_file)
    
#     Convert parse into sequences
    if log_source == "Linux":
        linux_seq(parsed_logs)
#     elif log_source == "HDFS":
#         hdfs_seq(parsed_logs, input_dir, log_source)
#     elif log_source == "Openstack":
# #         openstack_seq_instance(parsed_logs)
#         openstack_seq(output_dir, log_source)
    
    return
    
def linux_seq(df_log):
    
    df_log.Month = pd.to_datetime(df_log.Month, format='%b').dt.month
    df_log['Date'] = df_log['Date'].astype(str)
    df_log['Month'] = df_log['Month'].astype(str)    
    
    searchfor = ['jy|bt|mw']

    norm = df_log[~df_log.Level.str.contains('|'.join(searchfor), regex=True)]
    abnorm = df_log[df_log.Level.str.contains('|'.join(searchfor), regex=True)]
    
    normal = linux_time(norm)
    linux_file_generator("Linux", 'normal', normal)
    
    abnormal = linux_time(abnorm)
    linux_file_generator("Linux", 'abnormal', abnormal)    
    
    return

def linux_time(df):
#     df['datetime'] = pd.to_datetime(df['Time'])
    df['datetime'] = pd.to_datetime('2020' + "-" + df['Month'] + "-" + df['Date'] + " " + df['Time'])
    df = df[['datetime', 'Log Key']]
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df

def linux_file_generator(log_source, filename, df):
    with open("Dataset/" + log_source + "/" + log_source + "_" + filename, 'w') as f:
        for event_id_list in df['Log Key']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n') 

def openstack_seq_instance(df_log):
    normal = df_log[df_log["Date"] != "2017-05-14"]
    abnormal = df_log[df_log["Date"] == "2017-05-14"]
    normal_logs = normal.groupby("Instance")["Log Key"]
    abnormal_logs = abnormal.groupby("Instance")["Log Key"]
    
    with open( "Dataset/Openstack/openstack" + '_normal', 'w') as f:
        for name, group in normal_logs:
            s = " ".join(map(str, group.to_list()))
            f.write(s)
            f.write("\n")
    
    with open( "Dataset/Openstack/openstack" + '_abnormal', 'w') as f:
        for name, group in abnormal_logs:
            s = " ".join(map(str, group.to_list()))
            f.write(s)
            f.write("\n")        
    
    return
    
def hdfs_seq(df_log, output_dir, log_source):
    labels = pd.read_csv( "Dataset/" + log_source + "/anomaly_label.csv").groupby("Label")
    normal_labels = labels.get_group("Normal")["BlockId"].values
    anomaly_labels = labels.get_group("Anomaly")["BlockId"].values

    norm_seq = {}
    abn_seq = {}
    
    #Iterate through all parsed logs to create sequences
    for index, row in df_log.iterrows():
        # Get raw log content
        line= row['Content']

        # Block ids are can be in two different formats
        if re.search("blk_-\d*", line):
            seq_id = re.findall("blk_-\d*", line)[0]
        elif re.search("blk_\d*", line):
            seq_id = re.findall("blk_\d*", line)[0]
        else:
            print("Missing Block ID")

        if seq_id in normal_labels:
            if seq_id in norm_seq:
                norm_seq[seq_id].append(row['Log Key'])
            else:
                norm_seq[seq_id] = [row['Log Key']]
        if seq_id in anomaly_labels:
            if seq_id in abn_seq:
                abn_seq[seq_id].append(row['Log Key'])
            else:
                abn_seq[seq_id] = [row['Log Key']]

    hdfs_file_generator(output_dir, log_source, norm_seq, abn_seq)
    
    return
            
def openstack_seq(output_dir, log_source):
    df = pd.read_csv('Spell_results/openstack.log_structured.csv')
    df_train = df[df["Date"] == "2017-05-17"]
    df_normal = df[df["Date"] == "2017-05-16"]
    df_abnormal = df[df["Date"] == "2017-05-14"]

    log_source="Openstack"

    deeplog_test_normal = deeplog_df_transfer(df_normal)
    openstack_file_generator(log_source, 'train', deeplog_test_normal)

    deeplog_test_normal = deeplog_df_transfer(df_normal)
    openstack_file_generator(log_source, 'test_normal', deeplog_test_normal)    

    deeplog_test_abnormal = deeplog_df_transfer(df_abnormal)
    openstack_file_generator(log_source, 'test_abnormal', deeplog_test_abnormal)

    return

def hdfs_file_generator(input_dir, log_source, seqs, abn_seq = ''):
    with open(input_dir + log_source + '_normal', 'w') as f:
        for item in seqs:
            for log_key in seqs[item]:
                f.write(str(log_key)+" ")
            f.write("\n")
    with open(input_dir + log_source + '_abnormal', 'w') as f:
        for item in abn_seq:
            for log_key in abn_seq[item]:
                f.write(str(log_key)+" ")
            f.write("\n")

def pkl_to_csv(pkl_file, log_source, machine):
    with open('../System Logs/' + pkl_file, 'rb') as f:
        data = pickle.load(f)

    searchfor = ['filebeat\[(\d*)\]', 'packetbeat\[(\d*)\]', 'metricbeat\[(\d*)\]', 'auditbeat\[(\d*)\]']
    data = data[~data.message.str.contains('|'.join(searchfor), regex=True)]
    data = data.message
    
    file = pkl_file.split('.')[0]
    data.to_csv("Dataset/" + log_source + '/' + machine + '/' + file, index=False, header=False)
    
    return file

def federated_split(log_seq, log_source, clients):
    input_dir = "Dataset/" + log_source + '/'
    no_lines = sum(1 for line in open(input_dir + log_seq))
    chunk_size = no_lines//clients
    f_id = 1
    
    with open(input_dir + log_seq) as infile:
        f = open(input_dir + log_seq + '_%d' %f_id, 'w')
        
        for i, line in enumerate(infile):
            f.write(line)

            if not i % chunk_size and i != 0:
                f.close()
                f_id += 1
                if f_id <= clients:
                    f = open(input_dir + log_seq + '_%d' %f_id, 'w')
                else:
                    break
        f.close()
        
def backtrace(pred, log_source, algorithm):
    #log_template = pd.read_csv(algorithm + "_result/practicum_and_abnormal/" + log_source + "_templates.csv")
    log_template = pd.read_csv("../Source_Code/" + algorithm + "_result/CTDD/" + log_source + "_templates.csv")
#     y = np.squeeze(pred.tolist())
    
    for log in pred:
        if log == -1: continue
        print(log, log_template.loc[log_template['Log Key']==log]['Message'].to_string())
        
def deeplog_df_transfer(df):
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'Log Key']]
#     df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df

def _custom_resampler(array_like):
    return list(array_like)

def openstack_file_generator(log_source, filename, df):
    with open("Dataset/" + log_source + "/" + log_source + "_" + filename, 'w') as f:
        for event_id_list in df['Log Key']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')