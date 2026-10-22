import tensorflow as tf
import time

# Force execution on the GPU
with tf.device('/GPU:0'):
    # Create two large random matrices
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])

    start_time = time.time()
    # Perform matrix multiplication
    c = tf.matmul(a, b)
    end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")
print(f"Computed on device: {c.device}")