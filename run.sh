datas=("dataset/msong-1filter-80a" "dataset/deep1M-2filter-50a" "dataset/tiny5m-6filter-12a" "dataset/sift10m-6filter-6a")
num_codebooks=(256)
m=(4)

for num_codebook in ${num_codebooks[@]}; do
  for data in ${datas[@]}; do
    for subv_dim in ${m[@]}; do  

      sudo echo 3 | sudo tee /proc/sys/vm/drop_caches
      
      current_time=$(date "+%Y-%m-%d %H:%M:%S")
      echo "Current Time : $current_time , num_codebook: ${num_codebook}, subv_dim: ${subv_dim}, data: ${data} done"

      log_dir="Results/${data}_${num_codebook}_${subv_dim}.log"

      python -u src/filterPQ.py --num_codebook ${num_codebook} --subv_dim ${subv_dim} --data ${data} --gpu_num 8 > ${log_dir}
    done
  done
done
