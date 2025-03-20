from tests.t_generator import generate_test_class

generate_test_class("primitives.jnp", "jnp.add", globals())
generate_test_class("primitives.jnp", "jnp.concat", globals())
generate_test_class("primitives.jnp", "jnp.einsum", globals())
generate_test_class("primitives.jnp", "jnp.matmul", globals())
generate_test_class("primitives.jnp", "jnp.reshape", globals())
generate_test_class("primitives.jnp", "jnp.squeeze", globals())
generate_test_class("primitives.jnp", "jnp.tile", globals())
generate_test_class("primitives.jnp", "jnp.transpose", globals())
