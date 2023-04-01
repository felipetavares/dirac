# Dirac DSL

Dirac DSL is a [Dirac notation][dirac-notation] parser and interpreter.

# Macro System

The primary use of the Dirac DSL is through its macro system. For example, if we
want to generate the $\vert + \rangle$ state:

``` math
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}
```

We could sum the $\vert 0 \rangle$ and $\vert 1 \rangle$ states and divide by their norm using algebraic notation inside the `xdirac!` macro:

```rust
use dirac::xdirac as dirac;
use tensor::ToTensor;

fn main() {
    let state = dirac!((|0> + |1>) / ||0> + |1>|));

    // ...
}
```

Where `state` will then have the value:

    Tensor {
        data: [
            Complex {
                re: 0.7071067811865475,
                im: 0.0,
            },
            Complex {
                re: 0.7071067811865475,
                im: 0.0,
            },
        ],
        shape: (2, 1)
    }
    
## Operations

- `|0101>` - arbitrary length registers
- `|0> x |1>` - kronecker product
- `<0|1>` - inner product
- `|1><0|` - outer product
- `3|0>` - scalar operations (`+`, `-`, `*`, `/`)
- `| |0> |` - norm
- `|0>'` - conjugate transpose 
- `|0>'` - conjugate transpose 

# REPL

``` sh
❯ cd lib/dirac
❯ cargo run --release
   Compiling dirac v0.1.0 (/home/felipe/Development/q30d/lib/dirac)
    Finished release [optimized] target(s) in 0.40s
     Running `/home/felipe/Development/q30d/target/release/dirac`
(|0> - |1>) / | |0> - |1> | 
0.7071067811865475+0i
-0.7071067811865475+0i
```

[dirac-notation]: https://en.wikipedia.org/wiki/Dirac_notation
