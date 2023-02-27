
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --sign=True --sign_mask=True --sign_mask_p=0.01 --sign_alpha=0.00001 --name=A_FedSIGN_IID_CIFAR10
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --rsa=True --dprsa=True --dprsa_type=F --name=A_DPRSA_F_IID_CIFAR10 
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --rsa=True --dprsa=True --dprsa_type=G --name=A_DPRSA_G_IID_CIFAR10

# python FL_accuracy.py --round=4000 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --majority_vote=True --name=A_MajorityVote_IID_CIFAR10

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=A_FedAVG_IID_CIFAR10

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=A_KRUM_IID_CIFAR10 --robust_agg=True --robust_agg_type=krum --attack_num=10

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=A_FedAVG_CDP_CIFAR10 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=A_FedAVG_LDP_CIFAR10 --ldp=True --ldp_epsilon=20 --ldp_delta=0.00001 --ldp_clip=3


# MNIST
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.001 --sign_alpha=0.001 --name=A_FedSIGN_IID_MNIST 
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=A_DPRSA_F_IID_MNIST 
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=G --name=A_DPRSA_G_IID_MNIST

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_IID_MNIST
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_KRUM_IID_MNIST --robust_agg=True --robust_agg_type=krum --attack_num=10
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_CDP_IID_MNIST --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_LDP_IID_MNIST --ldp=True --ldp_epsilon=20 --ldp_delta=0.00001 --ldp_clip=3

# python FL_accuracy.py --round=4000 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024 --model=lanet --majority_vote=True --name=A_MajorityVote_IID_MNIST


# FMNIST
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --sign=True --sign_mask=True --sign_mask_p=0.001 --sign_alpha=0.001 --name=A_FedSIGN_IID_FMNIST 
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=F --name=A_DPRSA_F_IID_FMNIST 
# python FL_accuracy.py --round=1000 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --rsa=True --dprsa=True --dprsa_type=G --name=A_DPRSA_G_IID_FMNIST

# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_IID_FMNIST
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_KRUM_IID_FMNIST --robust_agg=True --robust_agg_type=krum --attack_num=10
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_CDP_FMNIST --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 
# python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_FedAVG_LDP_FMNIST --ldp=True --ldp_epsilon=20 --ldp_delta=0.00001 --ldp_clip=3

# python FL_accuracy.py --round=4000 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024 --model=lanet --majority_vote=True --name=A_MajorityVote_IID_FMNIST

# 借用跑NonIIDKrum
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=cifar10 --client_bs=1024 --client_lr=0.1 --model=alexnet --name=A_KRUM_NonIID_CIFAR10 --robust_agg=True --robust_agg_type=krum --attack_num=10
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=mnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_KRUM_NonIID_MNIST --robust_agg=True --robust_agg_type=krum --attack_num=10
python FL_accuracy.py --round=300 --client_ep=2 --data_distributed=noniid --update_type=weight --dataset=fmnist --client_bs=1024 --client_lr=0.1 --model=lanet --name=A_KRUM_NonIID_FMNIST --robust_agg=True --robust_agg_type=krum --attack_num=10

