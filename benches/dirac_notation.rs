use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dirac::dirac;
use tensor::{KroneckerProduct, Runtime, Tensor};

fn compiletime(c: &mut Criterion) {
    c.bench_function("comptime", |b| {
        b.iter(|| {
            let _state = black_box(dirac!(|10101> x |01010> x |10101> x |101>));
        })
    });
}

fn runtime(c: &mut Criterion) {
    let k0: Tensor = black_box(dirac!(|0>));
    let k1: Tensor = black_box(dirac!(|1>));

    c.bench_function("runtime", |b| {
        b.iter(|| {
            let a = vec![k1.clone(), k0.clone(), k1.clone(), k0.clone(), k1.clone()].prod();
            let b = vec![k0.clone(), k1.clone(), k0.clone(), k1.clone(), k0.clone()].prod();
            let c = vec![k0.clone(), k1.clone(), k0.clone()].prod();

            let _state = a.prod(&b).prod(&a).prod(&c);
        })
    });
}

criterion_group!(benches, compiletime, runtime);
criterion_main!(benches);
