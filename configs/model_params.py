import argparse

def ml_params():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    params = parser.parse_args()

    return params