use super::expression::Expression;
use crate::tensor::{AsTensor, KroneckerProduct, Tensor};
use nom::branch::alt;
use nom::bytes::complete::take_while1;
use nom::character::complete::char;
use nom::combinator::{all_consuming, opt};
use nom::multi::many0;
use nom::sequence::Tuple;
use nom::IResult;
use num::complex::Complex64;
use whitespace::ws;

mod whitespace;

// Builds a tensor from a sequence of 01+- characters by applying a sequence of
// Kronecker products.
fn tensor_basis(basis: &str) -> Tensor {
    basis
        .chars()
        .map(|c| c.as_tensor())
        .collect::<Vec<Tensor>>()
        .prod()
}

// Matches a string composed of 01+- representing a basis for a quantum state
fn basis(input: &str) -> IResult<&str, &str> {
    ws(take_while1(|c: char| "01+-".contains(c)))(input)
}

// Matches a ket |ket>
fn ket(input: &str) -> IResult<&str, Expression> {
    let (rem, (_, ket_str, _)) = (char('|'), basis, char('>')).parse(input)?;

    Ok((rem, Expression::Ket(tensor_basis(ket_str))))
}

// Matches a bra <bra|
fn bra(input: &str) -> IResult<&str, Expression> {
    let (rem, (_, bra_str, _)) = (char('<'), basis, char('|')).parse(input)?;

    Ok((rem, Expression::Bra(tensor_basis(bra_str))))
}

// Matches ehter the real or imaginary part of a complex number
fn number(input: &str) -> IResult<&str, Expression> {
    let (rem, num_str) = take_while1(|c: char| c.is_numeric() || c == '.' || c == 'i')(input)?;

    // NOTE: We are just unwraping the result of the string conversion so any
    // syntax errors while typing scalars are going to manifest themselves as
    // panics.

    // We are dealing with the imaginary part of a complex number...
    if num_str.ends_with('i') {
        if num_str.len() == 1 {
            Ok((rem, Expression::Scalar(Complex64::new(0.0, 1.0))))
        } else {
            Ok((
                rem,
                Expression::Scalar(Complex64::new(
                    0.0,
                    num_str[..num_str.len() - 1].parse().unwrap(),
                )),
            ))
        }
    }
    // ...and with the real part
    else {
        Ok((
            rem,
            Expression::Scalar(Complex64::new(num_str.parse().unwrap(), 0.0)),
        ))
    }
}

// Matches a parenthised expression ( expr )
fn parenthised(input: &str) -> IResult<&str, Expression> {
    let (rem, (_, expr, _)) = (char('('), additive, char(')')).parse(input)?;

    Ok((rem, Expression::Parenthised(Box::new(expr))))
}

// Matches a normalized expression | expr |
fn norm(input: &str) -> IResult<&str, Expression> {
    let (rem, (_, expr, _)) = (char('|'), additive, char('|')).parse(input)?;

    Ok((rem, Expression::Norm(Box::new(expr))))
}

// Matches a bra-ket inner product <bra|ket>
fn inner(input: &str) -> IResult<&str, Expression> {
    let (rem, (_, bra_str, _, ket_str, _)) =
        (char('<'), basis, char('|'), basis, char('>')).parse(input)?;

    Ok((
        rem,
        Expression::Inner(
            Box::new(Expression::Bra(tensor_basis(bra_str))),
            Box::new(Expression::Ket(tensor_basis(ket_str))),
        ),
    ))
}

// Matches a bra-ket outer product |ket><bra|
fn outer(input: &str) -> IResult<&str, Expression> {
    let (rem, ketbra) = (ket, bra).parse(input)?;

    match ketbra {
        (Expression::Ket(ket), Expression::Bra(bra)) => Ok((rem, Expression::Outer(ket, bra))),
        _ => unreachable!("ket and bra must return a ket and a bra"),
    }
}

// Matches one of:
// - scalar
// - outer bra-ket product
// - inner bra-ket product
// - bra
// - ket
// - parenthised expression
// - normalzied expression
fn atom(input: &str) -> IResult<&str, Expression> {
    alt((
        ws(number),
        ws(outer),
        ws(inner),
        ws(bra),
        ws(ket),
        ws(parenthised),
        ws(norm),
    ))(input)
}

// Matches a transpose conjugate operation in the form expr'
fn dag(input: &str) -> IResult<&str, Expression> {
    let (rem, out) = (atom, opt(char('\''))).parse(input)?;

    match out {
        (expr, Some(_)) => Ok((rem, Expression::Dagger(Box::new(expr)))),
        (expr, None) => Ok((rem, expr)),
    }
}

// Matches the additive inverse of some expression, or the expression itself: expr or -expr
fn inverse(input: &str) -> IResult<&str, Expression> {
    let (rem, (inverse, expr)) = (opt(char('-')), ws(dag)).parse(input)?;

    match (inverse, expr) {
        (Some(_), expr) => Ok((rem, Expression::AdditiveInverse(Box::new(expr)))),
        (None, expr) => Ok((rem, expr)),
    }
}

// Matches a multiplicative operation expr op expr, where op is one of *, /, x, .
// x represents the Kronecker product.
// . represents the dot (inner) product.
fn multiplicative(input: &str) -> IResult<&str, Expression> {
    let operation = |input| {
        let (rem, (char, expr)) =
            (alt((char('*'), char('/'), char('x'), char('.'))), inverse).parse(input)?;

        Ok((rem, (Some(char), expr)))
    };
    let direct = |input| {
        let (rem, expr) = ws(dag)(input)?;

        Ok((rem, (None, expr)))
    };
    let (rem, (first, rest)) = (inverse, many0(alt((operation, direct)))).parse(input)?;

    // Pass-through case: there are no operations so we just return the first
    // expression
    if rest.len() == 0 {
        return Ok((rem, first));
    }

    // Accumulator
    let mut acc = first;

    for operation in rest {
        match operation {
            (Some('*'), expr) => acc = Expression::Mul(Box::new(acc), Box::new(expr)),
            (Some('/'), expr) => acc = Expression::Div(Box::new(acc), Box::new(expr)),
            (Some('x'), expr) => acc = Expression::Kronecker(Box::new(acc), Box::new(expr)),
            (Some('.'), expr) => acc = Expression::Inner(Box::new(acc), Box::new(expr)),
            (Some(_), _) => unreachable!("should only ever match *, /, x, ."),
            (None, expr) => acc = Expression::Mul(Box::new(acc), Box::new(expr)),
        }
    }

    Ok((rem, acc))
}

// Matches additive expressions, sum or subtraction
fn additive(input: &str) -> IResult<&str, Expression> {
    let operation = |input| (alt((char('+'), char('-'))), multiplicative).parse(input);
    let (rem, (first, rest)) = (multiplicative, many0(operation)).parse(input)?;

    // Pass-through case: there are no operations so we just return the first
    // expression
    if rest.len() == 0 {
        return Ok((rem, first));
    }

    // Accumulator
    let mut acc = first;

    for operation in rest {
        match operation {
            ('+', expr) => acc = Expression::Add(Box::new(acc), Box::new(expr)),
            ('-', expr) => acc = Expression::Sub(Box::new(acc), Box::new(expr)),
            (_, _) => unreachable!("should only ever match +, -"),
        }
    }

    Ok((rem, acc))
}

// Matches a dirac notation expression
pub fn dirac(input: &str) -> IResult<&str, Expression> {
    all_consuming(additive)(input)
}

#[cfg(test)]
mod tests {
    use super::dirac;

    #[test]
    fn spaced() {
        assert!(dirac("| 0 > + | 1 >").is_ok());
    }

    #[test]
    fn debug() {
        let (_, expression) = dirac("<1| . |0> x |1> * (8 + 3i)").unwrap();
        dbg!(expression);
    }

    #[test]
    fn nested() {
        assert!(dirac("(0+(0+3)*5+(8+8*(3+2)))").is_ok());
    }

    #[test]
    fn numbers() {
        assert!(dirac("0*1").is_ok());
        assert!(dirac("1+1+100").is_ok());
        assert!(dirac("-12391.3").is_ok());
        assert!(dirac("1+8i").is_ok());
        assert!(dirac("-1-8i").is_ok());
    }

    #[test]
    fn spaces() {
        assert!(dirac("|0> x |0> + 3 * 7").is_ok());
    }

    #[test]
    fn kronecker() {
        assert!(dirac("|0>x|0>").is_ok());
    }

    #[test]
    fn outer() {
        assert!(dirac("|0><0|").is_ok());
        assert!(dirac("3*|0><0|").is_ok());
        assert!(dirac("|0><0|*3").is_ok());
        assert!(dirac("|0><0|+2").is_ok());
        assert!(dirac("(|0><0|+2)*3").is_ok());
        assert!(dirac("3*|0><0|*2").is_ok());
    }

    #[test]
    fn inner() {
        assert!(dirac("<0|0>").is_ok());
        assert!(dirac("3*<0|0>").is_ok());
        assert!(dirac("<0|0>/3").is_ok());
        assert!(dirac("<0|0>+|1>*<01|++>").is_ok());
        assert!(dirac("(<0|0>+<1|)*<01|++>").is_ok());
    }

    #[test]
    fn brakets() {
        assert!(dirac("|0>").is_ok());
        assert!(dirac("<0|").is_ok());
    }

    #[test]
    fn multiplicative() {
        assert!(dirac("|0>*|0>").is_ok());
        assert!(dirac("|0>*|0>*|0>").is_ok());
        assert!(dirac("|0>/|0>/|0>").is_ok());
        assert!(dirac("|0>/|0>*|0>").is_ok());

        assert!(dirac("|0>/|0>*").is_err());
    }

    #[test]
    fn additive() {
        assert!(dirac("|0>+|0>").is_ok());
    }

    #[test]
    fn parenthised() {
        assert!(dirac("(|0>)").is_ok());
        assert!(dirac("(|0>+|0>)").is_ok());
        assert!(dirac("(|0>*|0>)").is_ok());
        assert!(dirac("(|0>*|0>+|0>/|0>)").is_ok());

        assert!(dirac("(|0>*|0>)").is_ok());
        assert!(dirac("|0>*(|0>+|0>/|0>)").is_ok());
        // assert!(dirac_parser("1*(8+5+1)/(3+5)").is_ok());

        assert!(dirac("(").is_err());
        assert!(dirac(")").is_err());
        assert!(dirac("()").is_err());
        assert!(dirac("(|0>").is_err());
        assert!(dirac("|0>)").is_err());
    }

    #[test]
    fn inverse() {
        assert!(dirac("-|0>").is_ok());
        assert!(dirac("-<0|").is_ok());
        assert!(dirac("<0|-").is_err());
        assert!(dirac("|0>-").is_err());
    }

    #[test]
    fn mixed() {
        assert!(dirac("|0>+|0>-|1>/|1>").is_ok());
    }
}
