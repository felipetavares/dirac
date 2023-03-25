use crate::tensor::Tensor;
use num::complex::Complex64;

#[derive(Debug)]
pub enum Expression {
    Scalar(Complex64),

    Bra(Tensor),
    Ket(Tensor),

    AdditiveInverse(Box<Expression>),
    Dagger(Box<Expression>),

    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Kronecker(Box<Expression>, Box<Expression>),

    Inner(Box<Expression>, Box<Expression>),
    Outer(Tensor, Tensor),

    Parenthised(Box<Expression>),
    Norm(Box<Expression>),
}

impl Expression {
    pub fn compute(&self) -> Tensor {
        match self {
            Self::Scalar(c) => Tensor::new(vec![*c], (1, 1)),
            Self::Bra(bra) => bra.dag(),
            Self::Ket(ket) => ket.clone(),
            Self::AdditiveInverse(expr) => &expr.compute() * -1.,
            Self::Dagger(expr) => expr.compute().dag(),
            Self::Mul(a, b) => a.compute() * b.compute(),
            Self::Div(a, b) => a.compute() / b.compute(),
            Self::Add(a, b) => a.compute() + b.compute(),
            Self::Sub(a, b) => a.compute() - b.compute(),
            Self::Kronecker(a, b) => a.compute().prod(&b.compute()),
            Self::Inner(a, b) => Tensor::new(vec![a.compute() | b.compute()], (1, 1)),
            Self::Outer(a, b) => a * &b.dag(),
            Self::Parenthised(expr) => expr.compute(),
            Self::Norm(expr) => Tensor::new(vec![expr.compute().norm().into()], (1, 1)),
        }
    }
}
