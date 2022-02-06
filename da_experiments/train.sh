method=$1
source_data=$2
target_data=$3
seed=$4

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"


python -m train --base_config ${base_config_path} \
                --method_config ${method_config_path} \
                --opts source_data ${source_data} \
                    target_data ${target_data} \
                    visu True \
                    seed ${seed}