import jax

import jax.numpy as jnp

jax.distributed.initialize()

print(f"Total devices: {jax.device_count()}"," | " f"Devices per task: {jax.local_device_count()}")

x = jnp.ones(jax.local_device_count())

# Computes a reduction (sum) across all devices of x and broadcast the result, in y, to all devices.
# If x=[1] on all devices and we have 8 devices, the expected result is y=[8] on all devices.

y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(x)

print(f"Process {jax.process_index()} has y={y}")


