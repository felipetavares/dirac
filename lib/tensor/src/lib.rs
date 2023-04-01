//! Small CPU tensor library for compile-time macro tensor operations.

use num::complex::Complex64;
use std::{
    fmt::{self, Display},
    ops::{Add, BitOr, Div, Index, Mul, Sub},
};

type R = f64;
type C = Complex64;
type Data = Vec<C>;
type Shape = (usize, usize);

// Static tensor format for data transfer between compile time and runtime
type TensorData = (Shape, &'static [(f64, f64)]);

pub trait ToTensor {
    fn to_tensor(&self) -> Tensor;
}

impl ToTensor for TensorData {
    fn to_tensor(&self) -> Tensor {
        Tensor::new(
            self.1
                .iter()
                .map(|c| Complex64::new(c.0, c.1))
                .collect::<Vec<Complex64>>(),
            self.0,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Data,
    pub shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<C>, shape: Shape) -> Tensor {
        Tensor { data, shape }
    }

    pub fn eye(n: usize) -> Tensor {
        let mut data = Vec::<C>::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    data.push(C::new(1f64, 0f64));
                } else {
                    data.push(C::new(0f64, 0f64));
                }
            }
        }

        Tensor {
            data,
            shape: (n, n),
        }
    }

    pub fn item(&self) -> Option<C> {
        if self.shape != (1, 1) {
            return None;
        }

        Some(self.data[0])
    }

    pub fn norm_sqr(&self) -> R {
        self.data.iter().map(|c| c.norm_sqr()).sum()
    }

    pub fn norm(&self) -> R {
        self.norm_sqr().sqrt()
    }

    pub fn unit(&self) -> Tensor {
        self / self.norm()
    }

    // Dagger - conjugate transpose
    pub fn dag(&self) -> Tensor {
        match self.shape {
            (m, n) if m == 1 || n == 1 => {
                Tensor::new(self.data.iter().map(|c| c.conj()).collect(), (n, m))
            }
            (m, n) => {
                let mut data = Vec::<C>::new();

                for j in 0..n {
                    for i in 0..m {
                        data.push(self[(i, j)].conj());
                    }
                }

                Tensor::new(data, (n, m))
            }
        }
    }

    // Projector
    pub fn proj(&self) -> Tensor {
        self * &self.dag()
    }

    // Kronecker product
    pub fn prod(&self, rhs: &Tensor) -> Tensor {
        let shape = (self.shape.0 * rhs.shape.0, self.shape.1 * rhs.shape.1);
        let mut data = vec![C::new(0f64, 0f64); shape.0 * shape.1];

        // Walk the first matrix
        for i in 0..self.shape.1 {
            for j in 0..self.shape.0 {
                // For each element, walk the second matrix
                for k in 0..rhs.shape.1 {
                    for l in 0..rhs.shape.0 {
                        let x = i * rhs.shape.1 + k;
                        let y = j * rhs.shape.0 + l;

                        data[x + y * shape.1] = self[(j, i)] * rhs[(l, k)];
                    }
                }
            }
        }

        Tensor::new(data, shape)
    }

    pub fn expand(&self, n: usize, i: usize) -> Tensor {
        let eye = Tensor::eye(2);
        let mut product = if i == 0 { self.clone() } else { eye.clone() };

        for k in 1..n {
            product = if k == i {
                product.prod(self)
            } else {
                product.prod(&eye)
            }
        }

        product
    }
}

macro_rules! tensor_elementwise_op {
    ( $trait:ident, $op:ident ) => {
        impl $trait for Tensor {
            type Output = Tensor;

            fn $op(self, rhs: Tensor) -> Tensor {
                assert!(self.shape == rhs.shape);

                Tensor::new(
                    self.data
                        .iter()
                        .zip(rhs.data.iter())
                        .map(|(c1, c2)| c1.$op(c2))
                        .collect(),
                    self.shape,
                )
            }
        }
    };
}

impl Index<(usize, usize)> for Tensor {
    type Output = C;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.1 + index.0 * self.shape.1]
    }
}

tensor_elementwise_op!(Add, add);
tensor_elementwise_op!(Sub, sub);

impl Div<f64> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Tensor {
        Tensor::new(self.data.iter().map(|c| c / rhs).collect(), self.shape)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Tensor {
        Tensor::new(self.data.iter().map(|c| c * rhs).collect(), self.shape)
    }
}

impl Mul<C> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: C) -> Tensor {
        Tensor::new(self.data.iter().map(|c| c * rhs).collect(), self.shape)
    }
}

// Dot product
impl BitOr for Tensor {
    type Output = C;

    fn bitor(self, rhs: Tensor) -> C {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(c1, c2)| c1 * c2)
            .sum()
    }
}

// Matrix multiplication
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape.1 == rhs.shape.0 || self.shape == (1, 1) || rhs.shape == (1, 1));

        if self.shape == (1, 1) {
            return rhs * self.item().unwrap();
        }

        if rhs.shape == (1, 1) {
            return self * rhs.item().unwrap();
        }

        let shape = (self.shape.0, rhs.shape.1);
        let mut data = Vec::<C>::new();
        let n = self.shape.1;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                data.push((0..n).map(|k| self[(i, k)] * rhs[(k, j)]).sum());
            }
        }

        Tensor::new(data, shape)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        assert!(rhs.shape == (1, 1));

        Tensor::new(
            self.data.iter().map(|c| c / rhs.item().unwrap()).collect(),
            self.shape,
        )
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

/// Converts some type to a tensor
pub trait AsTensor {
    fn as_tensor(&self) -> Tensor;
}

impl AsTensor for char {
    fn as_tensor(&self) -> Tensor {
        match self {
            '0' => Tensor::new(vec![C::new(1.0, 0.0), C::new(0.0, 0.0)], (2, 1)),
            '1' => Tensor::new(vec![C::new(0.0, 0.0), C::new(1.0, 0.0)], (2, 1)),
            '+' => Tensor::new(vec![C::new(1.0, 0.0), C::new(1.0, 0.0)], (2, 1)).unit(),
            '-' => Tensor::new(vec![C::new(1.0, 0.0), C::new(0.0, -1.0)], (2, 1)).unit(),
            not_well_known => {
                panic!(
                    "Cannot decode '{}' into a qubit state: only (0, 1, +, -) supported",
                    not_well_known
                );
            }
        }
    }
}

pub trait KroneckerProduct {
    fn prod(&self) -> Tensor;
}

impl KroneckerProduct for Vec<Tensor> {
    fn prod(&self) -> Tensor {
        match self
            .iter()
            .cloned()
            .reduce(|product, tensor| product.prod(&tensor))
        {
            Some(tensor) => tensor,
            None => panic!("Should always be called on a nonempty vector"),
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.shape.0 {
            for x in 0..self.shape.1 {
                match x {
                    0 => write!(f, "{}", self[(y, x)])?,
                    _ => write!(f, ", {}", self[(y, x)])?,
                }
            }

            if y < self.shape.0 - 1 {
                writeln!(f, "")?;
            }
        }

        Ok(())
    }
}
