# Query the number of GPU devices such that each process per node has > 1 GPUs attached
import os
import jax, jax.numpy as jnp

jax.distributed.initialize(num_processes=int(os.environ['SLURM_NTASKS']),
                           local_device_ids=[x for x in range(int(os.environ['SLURM_GPUS_PER_NODE']))]
                           )

print(f"# Local devices: [ {jax.local_device_count()} ], {jax.local_devices()}")
print(f"# Global devices:[ {jax.device_count()},  {jax.devices()}")



