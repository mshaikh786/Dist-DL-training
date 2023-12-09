# Query the number of GPU devices such that each process per node has one GPU attached

import jax, jax.numpy as jnp

jax.distributed.initialize()

print(f"# Local devices: [ {jax.local_device_count()} ], {jax.local_devices()}")
print(f"# Global devices:[ {jax.device_count()},  {jax.devices()}")



