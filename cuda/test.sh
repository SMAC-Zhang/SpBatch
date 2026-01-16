# !/usr/bin/bash



############## test kernel spmm ##############
# M_values=(8 16 32 64)

# N_values=("10k" "15k" "20k" "30k" "40k")

# K_values=("4k" "8k" "12k")

# p_values=("95,85" "90,80" "90,75" "85,70", "80,60")

# for M in "${M_values[@]}"; do
#     for N_str in "${N_values[@]}"; do
#         N_val=$((${N_str%k} * 1024 ))
        
#         for K_str in "${K_values[@]}"; do
#             K_val=$((${K_str%k} * 1024 ))
            
#             for p_pair in "${p_values[@]}"; do
#                 IFS=',' read -r p1 p2 <<< "$p_pair"
                
#                 echo "Running with: M=$M, N=$N_val, K=$K_val, p1=$p1, p2=$p2"
#                 CUDA_VISIBLE_DEVICES=3 ./spmm_test kernel.csv $M $N_val $K_val $p1 $p2 10
                
#                 sleep 1
#             done
#         done
#     done
# done

# echo "All combinations completed!"


############## test ffn ##############
# for ((batch=8;batch<=64;batch=batch*2)) do
#     for layer in {0..31}; do
#         echo "Running with: layer=$layer, batch=$batch"
#         CUDA_VISIBLE_DEVICES=1 ./ffn_test ffn.csv $batch $layer 10
#         sleep 1
#     done
# done 

############## test ffn w/o division ##############
for ((batch=8;batch<=64;batch=batch*2)) do
    for layer in {0..31}; do
        echo "Running with: layer=$layer, batch=$batch"
        CUDA_VISIBLE_DEVICES=7 ./ffn_test cost.csv $batch $layer 10
        sleep 1
    done
done 