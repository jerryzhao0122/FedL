# python FL-privacy.py --white_box=True --round=301 --client_ep=5 --data_distributed=iid --dataset=cifar10 --client_bs=2048 --client_lr=0.01 --num_client=4 --train_num=4 --update_type=weight --model=alexnet --name=P_FedAVG

# python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=cifar10 --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG --white_box=True 

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=mnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG_MNIST --white_box=True 
python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=fmnist --client_bs=1024 --model=alexnet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG_FMNIST --white_box=True 

python FL-privacy.py --round=300 --client_ep=2 --data_distributed=iid --dataset=mnist --client_bs=1024 --model=lanet --num_client=4 --train_num=4 --update_type=weight --name=test_P_FedAVG_MNIST_lenet --white_box=True 