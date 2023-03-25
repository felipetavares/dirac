use super::parser;
use super::tensor::Tensor;
use num::complex::Complex64;

const EPSILON: f64 = 0.01;

fn compute_tensor(expression: &str) -> Tensor {
    parser::dirac(expression).unwrap().1.compute()
}

fn compute_complex(expression: &str) -> Complex64 {
    compute_tensor(expression).item().unwrap()
}

macro_rules! c {
    ($re:expr, $im:expr) => {
        Complex64::new($re, $im)
    };
    ($re:expr) => {
        Complex64::new($re, 0.0)
    };
}

#[test]
fn computation() {
    assert!((compute_complex("3(3)") - c![9.0]).norm() < EPSILON);
    assert!((compute_complex("2 + 2") - c![4.0]).norm() < EPSILON);
    assert!((compute_complex("2 / 2") - c!(1.0, 0.0)).norm() < EPSILON);
    assert!((compute_complex("(2 + 2) * 3") - c!(12.0, 0.0)).norm() < EPSILON);
    assert!((compute_complex("2 + 2 * 3") - c!(8.0, 0.0)).norm() < EPSILON);
    assert!((compute_complex("2 + 1i") - c!(2.0, 1.0)).norm() < EPSILON);
    assert!((compute_complex("1i + 2") - c!(2.0, 1.0)).norm() < EPSILON);
    assert!((compute_complex("3 * (1 + i)") - c!(3.0, 3.0)).norm() < EPSILON);
    assert!((compute_complex("3 * 1+i") - c!(3.0, 1.0)).norm() < EPSILON);

    assert!((compute_tensor("|0>") - Tensor::new(vec![c![1.0], c![0.0]], (2, 1))).norm() < EPSILON);
    assert!((compute_tensor("|1>") - Tensor::new(vec![c![0.0], c![1.0]], (2, 1))).norm() < EPSILON);

    assert!((compute_complex("<0|0>") - c![1.0]).norm() < EPSILON);
    assert!((compute_complex("<0|1>") - c![0.0]).norm() < EPSILON);

    assert!(
        (compute_tensor("|1><1|") - Tensor::new(vec![c![0.0], c!(0.0), c!(0.0), c![1.0]], (2, 2)))
            .norm()
            < EPSILON
    );

    assert!(
        (compute_tensor("|0><0|") - Tensor::new(vec![c![1.0], c!(0.0), c!(0.0), c!(0.0)], (2, 2)))
            .norm()
            < EPSILON
    );

    assert!(
        (compute_tensor("|0>3") - Tensor::new(vec![c!(3.0), c!(0.0)], (2, 1))).norm() < EPSILON
    );

    assert!(
        (compute_tensor("3|0>") - Tensor::new(vec![c!(3.0), c!(0.0)], (2, 1))).norm() < EPSILON
    );

    assert!((compute_tensor("|1> x |0>") - compute_tensor("|10>")).norm() < EPSILON);

    assert!((compute_tensor("|0>' - <0|").norm() < EPSILON));
    assert!((compute_tensor("|0> - <0|'").norm() < EPSILON));
    assert!((compute_tensor("<0|' - |0>").norm() < EPSILON));
    assert!((compute_tensor("<0| - |0>'").norm() < EPSILON));

    assert!((compute_complex("||1>|") - c![1.0]).norm() < EPSILON);

    assert!(compute_complex("||1>| - |<1||").norm() < EPSILON);
}
