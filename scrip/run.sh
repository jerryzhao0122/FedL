# noniid 正常实验，round=10，client=20，trainfrac=1
# python FedL.py --data_distributed=noniid_equal --round=10
# python FedL.py --data_distributed=noniid_unequal --round=10

# # noniid_equal 攻击实验，round=100，replace_attack_round=80
# python FedL.py --data_distributed=noniid_equal --round=100 --attack=True --attack_type=replace --attack_replace_round=80
# python FedL.py --data_distributed=noniid_equal --round=100 --attack=True --attack_type=pixel 
# # python FedL.py --data_distributed=noniid_equal --attack=True --attack_type=label

# # noniid_equal 防御实验，round=1500，replace_attack_round=1300
# python FedL.py --data_distributed=noniid_equal --rsa=True --round=1500
# python FedL.py --data_distributed=noniid_equal --rsa=True --round=1500 --attack=True --attack_type=replace --attack_replace_round=1300
# python FedL.py --data_distributed=noniid_equal --rsa=True --round=1500 --attack=True --attack_type=pixel 
# # python FedL.py --data_distributed=noniid_equal --rsa=True --round=1500 --attack=True --attack_type=label

# # noniid_unequal 攻击实验，round=100，replace_attack_round=80
# python FedL.py --data_distributed=noniid_unequal --round=100 --attack=True --attack_type=replace --attack_replace_round=80
# python FedL.py --data_distributed=noniid_unequal --round=100 --attack=True --attack_type=pixel 
# # python FedL.py --data_distributed=noniid_unequal --attack=True --attack_type=label

# # noniid_unequal 防御实验，round=1500，replace_attack_round=1300
# python FedL.py --data_distributed=noniid_unequal --rsa=True --round=1500
# python FedL.py --data_distributed=noniid_unequal --rsa=True --round=1500 --attack=True --attack_type=replace --attack_replace_round=1300
# python FedL.py --data_distributed=noniid_unequal --rsa=True --round=1500 --attack=True --attack_type=pixel 
# # python FedL.py --data_distributed=noniid_unequal --rsa=True --round=1500 --attack=True --attack_type=label

# python Fedl.py --data_distributed=iid --dataset=cifar10 

python FedL.py --round=500 --client_ep=15 --data_distributed=iid --update_type=dirction --onud=True --dataset=cifar10 --client_bs=512 --onud_server_alpha=0.001  --onud_client_lambda=0.001 --client_lr=0.02 --model=lanet --name=[ONUD-DPLR-sigma=5] --onud_sigma=5

python FedL.py --round=500 --client_ep=15 --data_distributed=iid --update_type=dirction --onud=True --dataset=cifar10 --client_bs=512 --onud_server_alpha=0.001  --onud_client_lambda=0.001 --client_lr=0.02 --model=lanet --name=[ONUD-DPLR-sigma=10] --onud_sigma=10

python FedL.py --round=500 --client_ep=15 --data_distributed=iid --update_type=dirction --onud=True --dataset=cifar10 --client_bs=512 --onud_server_alpha=0.001  --onud_client_lambda=0.001 --client_lr=0.02 --model=lanet --name=[ONUD-DPLR-sigma=15] --onud_sigma=15
