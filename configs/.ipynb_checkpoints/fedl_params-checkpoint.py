import argparse


def fl_params():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--name',type=str, default='normal',help='实验名字备注信息')
    parser.add_argument('--model', type=str, default='lanet', help='{resnet,lanet},model name')
    parser.add_argument('--kernel_num', type=int, default=9,help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    #Client params
    parser.add_argument('--num_client', type=int, default=20,help="number of users: K")
    parser.add_argument('--train_num', type=int, default=20,help='the fraction of clients: C')
    parser.add_argument('--update_type',type=str, default='weight',help='返回的类型:weight,dirction')
    parser.add_argument('--client_ep', type=int, default=4, help="the number of local epochs: E")
    parser.add_argument('--client_bs', type=int, default=64,help="local batch size: B")
    parser.add_argument('--client_lr', type=float, default=0.1,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,help='SGD momentum (default: 0.5)')

    #Server params
    parser.add_argument('--round', type=int, default=10,help='整体训练的轮数')
    parser.add_argument('--agg_type',type=str,default='aggWeight',help='服务器聚合类型')
    parser.add_argument('--server_lr',type=float,default=0.1,help='服务器聚合时候的学习率')

    #other params
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--data_distributed',type=str,default='iid',help='数据的分布情况: iid; non iid(dirichlet distribution); noniid_equal; noniid_unequal')
    parser.add_argument('--noniid_alpha',type=float,default=0.5,help='dirichlet distribution中的超参数')
    # parser.add_argument('--iid', type=int, default=1,help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--unequal', type=int, default=0,help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')      

    #client privacy LDP dp-sgd
    parser.add_argument('--ldp',type=bool,default=False,help='Using dp-sgd')
    parser.add_argument('--ldp_clip',type=float,default=1.0,help='Clip Grad')
    parser.add_argument('--ldp_sigma',type=float,default=0.1,help='')

    #Server Privacy# CDP Differentially Private arguments
    parser.add_argument('--cdp',type=bool,default=False,help="Whether train with DP")
    parser.add_argument('--sigma',type=float,default=2,help='sigma value')
    parser.add_argument('--epsilon',type=float,default=32,help='The epsilon for epsilon-delta privacy')
    parser.add_argument('--delta_budget',type=float,default=0.001,help='If out the delta_budger stop train')
    parser.add_argument('--delta',type=float,default=0.0001,help='The defined delta value')
    parser.add_argument('--order',type=float,default=32.0,help='I dont know what is order')
                 
    # RSA
    parser.add_argument('--rsa',type=bool,default=False,help='是否使用rsa方法')
    parser.add_argument('--alpha',type=float,default=0.001)
    parser.add_argument('--l1_lambda',type=float,default=0.07)
    parser.add_argument('--weight_lambda',type=float,default=0.01)
    
    # findAttacker
    parser.add_argument('--use_find',type=bool,default=False,help='是否开启识别恶意客户机功能')
    parser.add_argument('--cfdp_rho',type=float,default=0.05,help='截断距离pc的参数值')
    parser.add_argument('--cfdp_alpha',type=float,default=0.7,help='控制NMI的大小')
    parser.add_argument('--cfdp_beta',type=float,default=0.3,help='控制COS的大小')

    # ONUD
    parser.add_argument('--onud',type=bool,default=False,help='开启Only need update dirction')
    parser.add_argument('--onud_client_lambda',type=float,default=0.005,help='客户机端方向前面的参数')
    parser.add_argument('--onud_server_alpha',type=float,default=0.001,help='服务器聚合的sign前面的参数')
    parser.add_argument('--onud_root_dataset_numb',type=int,default=10,help='跟数据集每个类别的数据个数')
    parser.add_argument('--onud_root_data',type=bool,default=False,help='是否使用根数据')
    parser.add_argument('--onud_sigma',type=float,default=2,help='DP-LR中的参数')

    #Attack Backdoor
    parser.add_argument('--attack',type=bool,default=False,help='是否进行攻击')
    parser.add_argument('--attack_type',type=str,default='none',help='攻击的类型:replace,pixel,label,weightflip')
    parser.add_argument('--attack_num',type=int,default=6,help='恶意客户机的比例')

    #Attack Backdoor Replace
    parser.add_argument('--attack_replace_round',type=int,default=50,help='')
    parser.add_argument('--backdoor_replace_data_num',type=int,default=32,help='数据替换的数量')

    #Attack Backdoor Pixel 
    parser.add_argument('--pixel1',type=int,default=5,help='像素的行数')
    parser.add_argument('--pixel2',type=int,default=5,help='像素的列数')
    parser.add_argument('--pixel_target',type=int,default=7,help='表示加了标记的图片目标值')
    parser.add_argument('--pixel_attack_frequency',type=int,default=1,help='攻击频率,表示每几轮攻击一次')

    #Attack Backdoor LabelFlip
    parser.add_argument('--labelflip_original_label',type=int,default=7,help='表示原始的标签数值')
    parser.add_argument('--labelflip_target_label',type=int,default=2,help='表示反转后的标签')
    parser.add_argument('--labelflip_attack_frequency',type=int,default=2,help='每多少轮攻击一次')

    # Attack Weight Flip
    parser.add_argument('--weightflip_attack_frequency',type=int,default=1,help='多少轮攻击一次')

    # Attack Privacy White-Box 
    parser.add_argument('--white_box',type=str,default=False)

    params = parser.parse_args()
    
    return params
