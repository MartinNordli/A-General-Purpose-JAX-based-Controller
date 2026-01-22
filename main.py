import numpy as np
import jax
import jax.numpy as jnp

def jaxf1(x,y):
    q = x**2 + 8
    z = q**3 + 5*x*y

    return z

def jaxf2(x,y):
    z = 1
    for i in range(int(y)):
        z *= (x*float(i))

    return z

def jaxf3(x,y):
    return x**y

df3a = jax.grad(jaxf3, argnums=0)
df3b = jax.grad(jaxf3, argnums=1)
df3c = jax.grad(jaxf3, argnums=[0,1])

print(df3a(2.0, 3.0))
print(df3b(2.0, 3.0))
print(df3c(2.0, 3.0))
