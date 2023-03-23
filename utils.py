import random

def generate_random_abnormal_profile():
    """Generate random abnormal profiles"""
    a = random.randint(0, 23)
    b = random.uniform(0.00, 0.59)
    start_time_random = round(a+b, 2)
    connection_count_random = random.randint(40, 3000)
    most_frequent_IP_connection_count_random = random.randint(310, 20000)
    average_flow_duration_random = 0.0 # round(random.uniform(0.001, 5), 7)
    in_bytes_random = random.randint(10000, 2000000000)
    out_bytes_random = random.randint(10000, 2000000000)
    direct_ip_access_random = random.choice([0, 1])
    label  = -1

    return [connection_count_random, most_frequent_IP_connection_count_random, average_flow_duration_random,
            in_bytes_random, out_bytes_random, direct_ip_access_random, start_time_random, label]

def generate_random_normal_profile():
    """Generate normal training data function"""
    a = random.randint(7, 19)
    b = random.uniform(0.00, 0.59)
    start_time_random = round(a+b, 2)
    connection_count_random = random.randint(1, 9)
    most_frequent_IP_connection_count_random = random.randint(1, 10)
    average_flow_duration_random = 0.0 
    in_bytes_random = random.uniform(0.00, 300.0)
    out_bytes_random = random.uniform(0.00, 6000.0)
    direct_ip_access_random = 0
    label = 1

    return [connection_count_random, most_frequent_IP_connection_count_random, average_flow_duration_random,
            in_bytes_random, out_bytes_random, direct_ip_access_random, start_time_random, label]
