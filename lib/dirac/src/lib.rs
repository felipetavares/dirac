//! The `q30d` crate provides macros for generating qubits state using Dirac
//! notation.
//!
//! All register states generated using macros are computed at compile time.

extern crate proc_macro;
extern crate tensor;

use codegen::ToRust;
use proc_macro::TokenStream;
use std::str::FromStr;

mod codegen;
mod expression;
mod parser;

#[cfg(test)]
mod tests;

/// Generates a n-qubit quantum state from Dirac notation.
///
/// The standard ket notation can be used: `|01-+>`
/// The kronecker product ⊗ is represented by `x`
/// Standard tensor operations are supported: +, -, *, /
#[proc_macro]
pub fn dirac(input: TokenStream) -> TokenStream {
    let input_string = input.to_string();

    match parser::dirac(&input_string) {
        Ok((_, expression)) => {
            // Execute the expression
            let tensor = expression.compute();

            // Nothing we can do about stream errors at this point since this is
            // running inside the compiler, so we just unwrap.
            TokenStream::from_str(&tensor.to_rust("")).unwrap()
        }
        Err(e) => panic!(
            "Cannot interpret `{}` as dirac notation: {}",
            input_string, e
        ),
    }
}

/// Similar to the `dirac!` macro, but requires a runtime trait implementation
/// to convert the output to a runtime tensor format. The trait must implement
/// the `to_tensor()` function on the `TensorData` type which will be called by
/// the macro to convert the data transfer type to your custom tensor runtime
/// type.
///
/// ```
/// use tensor::TensorData;
///
/// trait ToTensor {
///   fn to_tensor(&self) -> CustomTensorType;
/// }
///
/// impl ToTensor for TensorData {
///   /* convert the tensor data transfer format
///      into an instance of CustomTensorType */
/// }
/// ```
///
/// This is similar to the extension trait pattern:
///
/// <https://rust-lang.github.io/rfcs/0445-extension-trait-conventions.html>
#[proc_macro]
pub fn xdirac(input: TokenStream) -> TokenStream {
    let input_string = input.to_string();

    match parser::dirac(&input_string) {
        Ok((_, expression)) => {
            // Execute the expression
            let tensor = expression.compute();

            // Nothing we can do about stream errors at this point since this is
            // running inside the compiler, so we just unwrap.
            TokenStream::from_str(&tensor.to_rust(".to_tensor()")).unwrap()
        }
        Err(e) => panic!(
            "Cannot interpret `{}` as dirac notation: {}",
            input_string, e
        ),
    }
}
