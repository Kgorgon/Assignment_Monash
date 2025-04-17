#!/bin/bash


MASTER_IP=$(kubectl get nodes -o wide | grep master | awk '{print $6}')

POD_COUNTS=(1 2 4 8)
MAX_USERS=100  # 初始最大用户数
TEST_DURATION=60  # 测试持续时间（秒）


mkdir -p experiment_results


for POD_COUNT in "${POD_COUNTS[@]}"; do
    echo "-----------------------------------"
    echo "Running experiment with $POD_COUNT pods"
    echo "-----------------------------------"
    
   
    kubectl scale deployment cloudpose-deployment --replicas=$POD_COUNT
    
 
    echo "Waiting for pods to be ready..."
    sleep 30
    
   
    kubectl get pods
    
    
    echo "Running load test on master node..."
    locust -f locustfile.py --host=http://$MASTER_IP:30080 --headless -u $MAX_USERS -r 10 --run-time ${TEST_DURATION}s --csv=experiment_results/master_${POD_COUNT}_pods
    

    
    echo "Experiment with $POD_COUNT pods completed"
    echo ""
done

echo "All experiments completed. Results are stored in the experiment_results directory."