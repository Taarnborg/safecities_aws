import argparse
import os
import pandas as pd
import csv

def create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac=.5):
    one_frac = 1-zero_frac

    zero_sample_size = int((n_train+n_valid)*zero_frac)
    one_sample_size = int((n_train+n_valid)*one_frac)
    n_train_zero = int(n_train/(n_train+n_valid)*zero_sample_size)
    n_train_one = int(n_train/(n_train+n_valid)*one_sample_size)

    while (zero_sample_size+one_sample_size) != n_train+n_valid:
        if zero_sample_size+one_sample_size < n_train+n_valid:
            zero_sample_size += 1
            n_train_zero += 1
        if zero_sample_size+one_sample_size > n_train+n_valid:
            zero_sample_size = zero_sample_size-1
            n_train_zero = n_train_zero -1

    sub_zeroes = df_zeroes.sample(n=zero_sample_size,random_state=args.seed)
    sub_ones = df_ones.sample(n=one_sample_size,random_state=args.seed)

    df_train = pd.concat([
        sub_zeroes[:n_train_zero],
        sub_ones[:n_train_one],
    ]).sample(frac=1,random_state=args.seed)
    df_valid = pd.concat([
        sub_zeroes[n_train_zero:],
        sub_ones[n_train_one:],
    ]).sample(frac=1,random_state=args.seed)

    assert bool(set(df_train.index.tolist()) & set(df_valid.index.tolist())) == False
    return df_train,df_valid

def main(args):

    if args.medborger:
        train_path = os.path.join(args.data_dir,'angreb_70.csv')
        df = pd.read_csv(train_path, sep='\t', names = ['targets', 'text'])
        df_zeroes = df.loc[df.targets == 0]
        df_ones = df.loc[df.targets == 1]

        # Create balanced train/valid set with 64/32 examples
        n_train = 64
        n_valid = 32
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_mini_balanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_mini_balanced_valid.csv'), sep='\t', header=False)
        # Create balanced train/valid set with 128/64 examples
        n_train = 128
        n_valid = 64
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_small_balanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_small_balanced_valid.csv'), sep='\t', header=False)
        # Create balanced train/valid set with 256/64 examples
        n_train = 256
        n_valid = 128
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_medium_balanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_medium_balanced_valid.csv'), sep='\t', header=False)
        # Create balanced train/valid set with 1024/128 examples
        n_train = 1024
        n_valid = 256
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_large_balanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_large_balanced_valid.csv'), sep='\t', header=False)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 128
        n_valid = 64
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_small_imbalanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_small_imbalanced_valid.csv'), sep='\t', header=False)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 256
        n_valid = 128
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_medium_imbalanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_medium_imbalanced_valid.csv'), sep='\t', header=False)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 1024
        n_valid = 256
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'medborger_large_imbalanced_train.csv'), sep='\t', header=False)
        df_valid.to_csv(os.path.join(args.data_dir,'medborger_large_imbalanced_valid.csv'), sep='\t', header=False)

    if args.safecities:
        print('safecities')

        train_path = os.path.join(args.data_dir,'hateful_70.csv')
        df = pd.read_csv(train_path, sep='\t', names = ['targets', 'text'])
        df_zeroes = df.loc[df.targets == 0]
        df_ones = df.loc[df.targets == 1]

        # Create balanced train/valid set with 64/32 examples
        n_train = 64
        n_valid = 32
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_mini_balanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_mini_balanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create balanced train/valid set with 128/64 examples
        n_train = 128
        n_valid = 64
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_small_balanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_small_balanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create balanced train/valid set with 256/64 examples
        n_train = 256
        n_valid = 128
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_medium_balanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_medium_balanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create balanced train/valid set with 1024/128 examples
        n_train = 1024
        n_valid = 256
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_large_balanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_large_balanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 128
        n_valid = 64
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_small_imbalanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_small_imbalanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 256
        n_valid = 128
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_medium_imbalanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_medium_imbalanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        # Create imbalanced train/valid set with 128/64 examples
        n_train = 1024
        n_valid = 256
        zero_frac = .8
        df_train,df_valid = create_dataset(df_zeroes,df_ones,n_train,n_valid,zero_frac)
        df_train.to_csv(os.path.join(args.data_dir,'safecities_large_imbalanced_train.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)
        df_valid.to_csv(os.path.join(args.data_dir,'safecities_large_imbalanced_valid.csv'), sep='\t', header=False,quoting=csv.QUOTE_ALL)



# df_train = pd.read_csv('angreb/train.csv', sep='\t', names = ['targets', 'text'])
# df_valid = pd.read_csv('angreb/valid.csv', sep='\t', names = ['targets', 'text'])
# test_word = 'idiot'
# max_size = 100
# subset_1 = df_train[(df_train.text.str.contains(test_word))&(df_train.targets == 1)].iloc[0:max_size]
# subset_0 = df_train[(~df_train.text.str.contains(test_word))&(df_train.targets == 0)].iloc[0:max_size]
# df_mini_train = pd.concat([subset_1, subset_0])
# df_mini_train.to_csv('angreb/mini/train.csv', sep='\t', header=False)
# subset_1 = df_valid[(df_valid.text.str.contains(test_word))&(df_valid.targets == 1)].iloc[0:max_size]
# subset_0 = df_valid[(~df_valid.text.str.contains(test_word))&(df_valid.targets == 0)].iloc[0:max_size]
# df_mini_valid = pd.concat([subset_1, subset_0])
# df_mini_valid.to_csv('angreb/mini/valid.csv', sep='\t', header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--medborger', action='store_false')
    parser.add_argument('--safecities', action='store_false')
    parser.add_argument("--seed", type=int, default=16032311)
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])

    ## RUN
    args = parser.parse_args()
    main(args)