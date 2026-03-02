while ($true) {
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits
    Start-Sleep -Seconds 2
}
