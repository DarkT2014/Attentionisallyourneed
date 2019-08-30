#-*-coding:utf-8-*-
from __future__ import division
import sys
import os
import io
import time

_package_path = "/".join(os.path.abspath(os.path.dirname(__file__)).split("/")[:-1])
sys.path.append(_package_path)

import tensorflow as tf
import jieba
from utils.tokenizer import _save_vocab_file, _load_vocab_file



MIN_COUNT = 5

_TRAIN_DATA = {
    'inputs': 'train.en',
    'targets': 'train.zh'
}


_DEV_DATA = {
    # 'inputs': 'dev.en',
    # 'targets': 'dev.zh'
    'inputs': 'newstest2017.en',
    'targets': 'newstest2017.zh'
}




def iterator_file(file_path):
    with io.open(file_path, encoding='utf-8') as inf:
        for i, line in enumerate(inf):
            yield i,line


_PREFIX = "translate"

def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))

def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in dictionary.items():
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))

# from transformer import textencoder

from transformer.utils import tokenizer

def create_tf_record(source_files, vocab_files, out_dir, mode, total_shards):
    input_encoder = tokenizer.Subtokenizer(vocab_files[0])
    target_encoder = tokenizer.Subtokenizer(vocab_files[1])
    shard_files = [shard_filename(out_dir, mode, n + 1, total_shards) for n in range(total_shards)]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in shard_files]

    input_file = source_files[0]
    target_file = source_files[1]
    counter = 0
    shard = 0
    for input_tuple, target_tuple in zip(iterator_file(input_file), iterator_file(target_file)):
        counter += 1

        input_line = input_tuple[1]
        target_line = target_tuple[1]
        
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)
            print("\tSaving case %d." % counter)
        # print("input line :" + input_line)
        # print("target line : " + target_line)
        example_dict = {
            'inputs':  input_encoder.encode(input_line[1], True),
            'targets': target_encoder.encode(target_line, True)
        }

        example = dict_to_example(example_dict)
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards


def write_file(outf, content_list):
    for line in content_list:
        outf.write(line + "\n")



if __name__ == '__main__':

    # source_dir = "./test_data"
    source_dir = "/tmp/t2t_datagen"


    ##### create vocab
    zh_vocab = os.path.join(source_dir, "vocab.translate_enzh_wmt32k.32768.subwords.zh")
    en_vocab = os.path.join(source_dir, "vocab.translate_enzh_wmt32k.32768.subwords.en")
    print("create vocab ...")



    en_source_file = os.path.join(source_dir, _TRAIN_DATA['inputs'])
    zh_source_file = os.path.join(source_dir, _TRAIN_DATA['targets'])


    inputs_tokenizer = tokenizer.Subtokenizer.init_from_files(
        en_vocab, [en_source_file],  2**15, 20,
        min_count=None, file_byte_limit=1e8)
    print("en tokenize done.")
    # targets_tokenizer = tokenizer.Subtokenizer.init_from_files(
    #     zh_vocab, [zh_source_file],  2**15, 20,
    #     min_count=None, file_byte_limit=1e8)
    zh_subtoken_list = []
    doIt = True
    if os.path.exists(zh_vocab):
        res = raw_input("Detected zh vocab file, skip it?(Y/N)")
        if res.lower() == 'n':
            doIt = True
        elif res.lower() == 'y':
            doIt = False
        elif res == '':
            doIt = False
    else:
        doIt = True          


    if doIt:
        zh_subtoken_list = ['<PAD>', '<GO>', '<EOS>']
        totalLineNum = os.popen('wc -l ' + zh_source_file).read().split()[0]
        global start_time
        start_time = time.time()
        # jieba.enable_parallel(4)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dict_path = os.path.join(dir_path, 'mydict.txt')
        print("path of user dict of jieba is : " + dict_path)
        jieba.load_userdict(dict_path)
        last_time = time.time()
        for i, line in iterator_file(zh_source_file):
            line = line.replace("\r","").replace("\n","")
            if(time.time() - last_time > 0.1):#print per 0.1sec
                last_time = time.time()
                percent = (i+1)/int(totalLineNum) * 100
                duration = time.time() - start_time
                sys.stdout.write("\r%.2f%% cutting %dth line of %d, %d sec passed. "% (percent, i+1, int(totalLineNum), duration))
                # print("\rcutting " + str(i) + "th line of :" + line, end='', flush=True)
                sys.stdout.flush()
            zh_subtoken_list.extend(jieba.lcut(line))
        print("\n Cut list done.")
    else:
        print("reading existed zh vocab file ...")
        with io.open(zh_vocab, encoding='utf8') as f:
            for line in f:
                zh_subtoken_list.append(line)
        print("done.")
    
    print("--------------------------------")
    
    
    print("Tokenizinig zh vocab ...")
    # targets_tokenizer = tokenizer.Subtokenizer.init_from_files(
    #     zh_vocab, [zh_source_file],  2**15, 20,
    #     min_count=None, file_byte_limit=1e9)
    print("Saving zh vocab files ...")
    _save_vocab_file(zh_vocab, zh_subtoken_list)
    # print("tokenizing zh vocab..")
    # targets_tokenizer = tokenizer.Subtokenizer(zh_vocab, zh_subtoken_list)


    print("Tokenizing zh vocab done.")


    # data_dir = './train_data'
    data_dir = '/tmp/t2t_datagen/'

    print("\nBegin to creating train_tfrecord\n")

    create_tf_record([en_source_file, zh_source_file], [en_vocab, zh_vocab], data_dir, 'train', 10 )

    
    create_tf_record([os.path.join(source_dir, _DEV_DATA['inputs']), os.path.join(source_dir, _DEV_DATA['targets'])],
                     [en_vocab, zh_vocab],
                     data_dir,
                     'dev',
                     1)
