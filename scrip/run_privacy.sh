python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedSIGN --onud=True --update_type=dirction 
python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedSIGN_CDP --onud=True --update_type=dirction --cdp=True --sigma=0.01
python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg
# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --update_type=weight --name=P_FedAvg_LDP --ldp=True  
python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.001 --num_client=4 --train_num=4 --name=P_FedSIGN --sign=True

python FL-privacy.py --white_box=True --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --model=alexnet --client_bs=1024 --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAvg
python FL-privacy.py --white_box=True --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --model=alexnet --client_bs=1024 --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAvg_LDP_5 --ldp=True --ldp_epsilon=5 --ldp_delta=0.00001 --ldp_clip=3 
python FL-privacy.py --white_box=True --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --model=alexnet --client_bs=1024 --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAvg_CDP_20 --cdp=True --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5

python FL-privacy.py --white_box=True --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --model=alexnet --client_bs=1024 --num_client=4 --train_num=4 --update_type=direction --name=test_P_FedSIGNV2 --sign_v2=True --sign_alpha=0.0001
python FL-privacy.py --white_box=True --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --model=alexnet --client_bs=1024 --num_client=4 --train_num=4 --update_type=direction --name=test_P_FedSIGN --sign=True --sign_alpha=0.0001

python FL-privacy.py --round=300 --client_ep=5 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=2048 --model=alexnet --client_lr=0.01 --num_client=4 --train_num=4 --name=test_P_FedSIGN_2 --sign=True --sign_alpha=0.0001 --white_box=True

python FL-privacy.py --round=300 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=2048 --model=alexnet --client_lr=0.01 --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_20 --cdp=True --cdp_epsilon=20 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True
python FL-privacy.py --round=300 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=2048 --model=alexnet --client_lr=0.01 --num_client=4 --train_num=4 --name=test_P_FedAvg_LDP_5 --ldp=True --ldp_epsilon=5 --ldp_delta=0.00001 --ldp_clip=3 --white_box=True


python FL-privacy.py --round=300 --client_ep=5 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=2048 --model=alexnet --client_lr=0.01 --num_client=4 --train_num=4 --name=test_P_FedAVG --white_box=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=F --name=test_DPRSA_F_iid_white_box --white_box=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=G --name=test_DPRSA_G_iid_white_box --white_box=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=F --name=test_DPRSA_F_iid_white_box_2 --white_box=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=G --name=test_DPRSA_G_iid_white_box_2 --white_box=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=direction --name=test_P_FedSIGN --sign=True --sign_alpha=0.0001

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG --white_box=True 

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_LDP_10 --ldp=True --ldp_epsilon=10 --ldp_delta=0.00001 --ldp_clip=3 --white_box=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=weight --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --name=test_P_FedAvg_CDP_10 --cdp=True --cdp_epsilon=10 --cdp_delta=0.00001 --cdp_clip=0.5 --white_box=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=F --name=test_DPRSA_F_MNIST --white_box=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=mnist --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=G --name=test_DPRSA_G_MNIST --white_box=True

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=F --name=test_DPRSA_F_FMNIST --white_box=True
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --update_type=direction --dataset=fmnist --client_bs=1024  --model=alexnet --num_client=4 --train_num=4 --rsa=True --dprsa=True --dprsa_type=G --name=test_DPRSA_G_FMNIST --white_box=True