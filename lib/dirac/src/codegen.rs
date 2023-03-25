use num::complex::Complex64;
use tensor::Tensor;

pub trait ToRust {
    fn to_rust(&self, suffix: &str) -> String;
}

impl ToRust for Tensor {
    fn to_rust(&self, suffix: &str) -> String {
        format!(
            "({}, {}){}",
            self.shape.to_rust(""),
            self.data.to_rust(""),
            suffix,
        )
    }
}

impl ToRust for &Complex64 {
    fn to_rust(&self, _: &str) -> String {
        format!("({}f64, {}f64)", self.re, self.im)
    }
}

impl ToRust for Vec<Complex64> {
    fn to_rust(&self, _: &str) -> String {
        format!(
            "&[{}][..]",
            self.iter()
                .map(|c| c.to_rust(""))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl ToRust for (usize, usize) {
    fn to_rust(&self, _: &str) -> String {
        format!("({}usize, {}usize)", self.0, self.1)
    }
}
