method=$1
seeds="0 1 2"
for seed in ${seeds}
do
    bash train.sh ${method} MNIST MNISTM ${seed}
    bash train.sh ${method} MNIST USPS ${seed}
    bash train.sh ${method} USPS MNIST ${seed}
    bash train.sh ${method} SVHN MNIST ${seed}
    bash train.sh ${method} CIFAR10 STL10 ${seed}
    bash train.sh ${method} STL10 CIFAR10 ${seed}
done