extern crate tensor;

use std::io::{self, BufRead};
use tensor::Tensor;

mod expression;
mod parser;

fn calculate(expression: &str) -> Result<Tensor, nom::Err<nom::error::Error<&str>>> {
    match parser::dirac(expression) {
        Ok((_, ast)) => Ok(ast.compute()),
        Err(e) => Err(e),
    }
}

fn main() {
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        match line {
            Err(e) => panic!("reading line: {:?}", e),
            Ok(line_str) => match calculate(&line_str) {
                Ok(tensor) => println!("{}", tensor),
                Err(e) => println!("Cannot interpret `{}` as dirac notation: {}", line_str, e),
            },
        }
    }
}
