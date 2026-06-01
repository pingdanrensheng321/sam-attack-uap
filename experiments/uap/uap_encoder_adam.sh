style='understand_cosine_sim'  #please input your style #'enc_rm_inf' #enc_rm_inf_2aug #understand_cosine_sim
current_path=$(dirname "$(readlink -f "$0")")
current_file=$(basename "$0")
echo "Current script file name: $current_file"
seed_list=(10) # 20 30 40 50) # 20 30 40 50)
step_nums=(1)
#############choose hyperparameter#############
if [ "$style" == "enc_rm_inf" ]; then
    grid_size_list=(10)
    attack_type='enc_rm_inf'
    train_img_num=100
    test_img_num=100
    neg_num=100 #contrasive loss
    echo "Use enc_rm_inf type"
elif [ "$style" == "enc_rm_inf_2aug" ]; then
    grid_size_list=(10)
    attack_type='enc_rm_inf_2aug'
    train_img_num=100
    test_img_num=100
    neg_num=100 #contrasive loss
    echo "Use enc_rm_inf type"
elif [ "$style" == "understand_cosine_sim" ]; then
    grid_size_list=(10)
    attack_type='understand_cosine_sim'
    train_img_num=100
    test_img_num=100
    neg_num=100 #contrasive loss
    echo "Use enc_rm_inf type"
else
    echo "Input Style is Error"
    exit 1
fi

for num in ${grid_size_list[*]};
do 
    for attack_num in ${step_nums[*]};
    do 
        for seed in ${seed_list[*]};
        do
        CUDA_VISIBLE_DEVICES=3 python evaluation/eval_attack_uap_cl.py \
            --experiment 'attack_uap_cl'\
            --attack_type $attack_type \
            --attack_lr 0.005\
            --point_number_size $num \
            --attack_init 'zero' \
            --random_point \
            --attack_steps $attack_num \
            --multimask_output_off \
            --sh_file_name $current_file\
            --train_img_num $train_img_num\
            --test_img_num $test_img_num\
            --train_uap \
            --eps 10 \
            --temperature 0.1 \
            --show_img 0 \
            --debug 1\
            --neg_num $neg_num\
            --sh_file_name $current_file\
            --seed $seed
        done
    done 
done 