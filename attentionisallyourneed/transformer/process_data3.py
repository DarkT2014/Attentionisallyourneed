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
from utils.tokenizer import _save_vocab_file
import random


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
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"
_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1

RAW_DIR = "/tmp/t2t_datagen/"
DATA_DIR = "/tmp/t2t_datagen/"

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


def compile_files(raw_dir, raw_files, tag):
    """Compile raw files into a single file for each language.

    Args:
    raw_dir: Directory containing downloaded raw files.
    raw_files: Dict containing filenames of input and target data.
        {"inputs": list of files containing data in input language
        "targets": list of files containing corresponding data in target language
        }
    tag: String to append to the compiled filename.

    Returns:
    Full path of compiled input and target files.
    """
    tf.logging.info("Compiling files with tag %s." % tag)
    filename = "%s-%s" % (_PREFIX, tag)
    input_compiled_file = os.path.join(raw_dir, filename + ".lang1")
    target_compiled_file = os.path.join(raw_dir, filename + ".lang2")

    with tf.gfile.Open(input_compiled_file, mode="w") as input_writer:
        with tf.gfile.Open(target_compiled_file, mode="w") as target_writer:
            for i in range(len(raw_files["inputs"])):
                input_file = raw_files["inputs"][i]
                target_file = raw_files["targets"][i]

                tf.logging.info("Reading files %s and %s." % (input_file, target_file))
                write_file(input_writer, input_file)
                write_file(target_writer, target_file)
    return input_compiled_file, target_compiled_file


def all_exist(filepaths):
    """Returns true if all files in the list exist."""
    for fname in filepaths:
        if not tf.gfile.Exists(fname):
            return False
    return True



def txt_line_iterator(path):
    """Iterate through lines of file."""
    with tf.gfile.Open(path) as f:
        for line in f:
            yield line.strip()


def shuffle_records(fname):
  """Shuffle records in a single file."""
  tf.logging.info("Shuffling records in file %s" % fname)

  # Rename file prior to shuffling
  tmp_fname = fname + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.python_io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        tf.logging.info("\tWriting record: %d" % count)

  tf.gfile.Remove(tmp_fname)

def encode_and_save_files(
    subtokenizer, data_dir, raw_files, tag, total_shards):
    """Save data from files as encoded Examples in TFrecord format.

    Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
        the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

    Returns:
    List of all files produced.
    """
    # Create a file for each shard.
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
                for n in range(total_shards)]

    if all_exist(filepaths):
        tf.logging.info("Files with tag %s already exist." % tag)
        return filepaths

    tf.logging.info("Saving files with tag %s." % tag)
    input_file = raw_files[0]
    target_file = raw_files[1]

    # Write examples to each shard in round robin order.
    tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
    counter, shard = 0, 0
    for counter, (input_line, target_line) in enumerate(zip(
            txt_line_iterator(input_file), txt_line_iterator(target_file))):
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)
        example = dict_to_example(
            {"inputs": subtokenizer.encode(input_line, add_eos=True),
             "targets": subtokenizer.encode(target_line, add_eos=True)})
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards
    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)

    tf.logging.info("Saved %d Examples", counter + 1)
    return filepaths




if __name__ == '__main__':

    # source_dir = "./test_data"
    source_dir = "/tmp/t2t_datagen"


    ##### create vocab
    zh_vocab = os.path.join(source_dir, "zh_sub_word.vocab")
    en_vocab = os.path.join(source_dir, "en_sub_word.vocab")
    print("create vocab ...")



    en_source_file = os.path.join(source_dir, _TRAIN_DATA['inputs'])
    zh_source_file = os.path.join(source_dir, _TRAIN_DATA['targets'])


    inputs_tokenizer = tokenizer.Subtokenizer.init_from_files(
        en_vocab, [en_source_file],  2**15, 20,
        min_count=None, file_byte_limit=1e8)
    print("en tokenize done.")
    




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
        # zh_subtoken_list = ['<PAD>', '<GO>', '<EOS>']
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
                zh_subtoken_list.append(line.replace("'","").strip('\n'))
        print("done.")
    
    print("--------------------------------")
    

    subtoken_list = []
    # with io.open(en_vocab, encoding='utf8') as f:
    #     for line in f:
    #         if line is not "'<pad>'\n" and line is not "'<EOS>'\n":
    #             subtoken_list.append(line.replace("'","").strip('\n'))
    subtoken_list = _load_vocab_file(en_vocab)


    print(subtoken_list[:10])
    print(zh_subtoken_list[:10])
    subtoken_list = subtoken_list + zh_subtoken_list
    
    vocab_file = os.path.join(DATA_DIR, "enzh.vocab")
    print("Tokenizinig zh vocab ...")
    # targets_tokenizer = tokenizer.Subtokenizer.init_from_files(
    #     zh_vocab, [zh_source_file],  2**15, 20,
    #     min_count=None, file_byte_limit=1e9)
    print("Saving zh vocab files ...")
    # _save_vocab_file(zh_vocab, zh_subtoken_list)
    _save_vocab_file(vocab_file, subtoken_list)



    print("Tokenizing zh vocab done.")


    # data_dir = './train_data'
    data_dir = '/tmp/t2t_datagen/'


    
    # print("\nBegin to creating train_tfrecord\n")

    # create_tf_record([en_source_file, zh_source_file], [en_vocab, zh_vocab], data_dir, 'train', 10 )

    
    # create_tf_record([os.path.join(source_dir, _DEV_DATA['inputs']), os.path.join(source_dir, _DEV_DATA['targets'])],
    #                  [en_vocab, zh_vocab],
    #                  data_dir,
    #                  'dev',
    #                  1)


    train_files = {
        'inputs': ['/tmp/t2t_datagen/train.en'],
        'targets': ['/tmp/t2t_datagen/train.zh']
    }
    eval_files = {
        'inputs': ['/tmp/t2t_datagen/newstest2017.en'],
        'targets': ['/tmp/t2t_datagen/newstest2017.zh']
    }

    subtokenizer = tokenizer.Subtokenizer(vocab_file)

    print("Step 3/4: Compiling training and evaluation data")
    compiled_train_files = compile_files(RAW_DIR, train_files, _TRAIN_TAG)
    compiled_eval_files = compile_files(RAW_DIR, eval_files, _EVAL_TAG)

  # Tokenize and save data as Examples in the TFRecord format.
    print("Step 4/4: Preprocessing and saving data")
    train_tfrecord_files = encode_and_save_files(
        subtokenizer, DATA_DIR, compiled_train_files, _TRAIN_TAG,
        _TRAIN_SHARDS)
    encode_and_save_files(
        subtokenizer, DATA_DIR, compiled_eval_files, _EVAL_TAG,
        _EVAL_SHARDS)

    for fname in train_tfrecord_files:
        shuffle_records(fname)
