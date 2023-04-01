use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dirac::{dirac, xdirac};
use tensor::{KroneckerProduct, Tensor, ToTensor};

fn compiletime_nocopy(c: &mut Criterion) {
    c.bench_function("comptime-nocopy", |b| {
        b.iter(|| {
            let _state = black_box(dirac!(|0000000000>));
        })
    });
}

fn compiletime(c: &mut Criterion) {
    c.bench_function("comptime", |b| {
        b.iter(|| {
            let _state = black_box(xdirac!(|0000000000>));
        })
    });
}

fn runtime(c: &mut Criterion) {
    let k0: Tensor = black_box(xdirac!(|0>));

    c.bench_function("runtime", |b| {
        b.iter(|| {
            let _state = vec![
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
                k0.clone(),
            ]
            .prod();
        })
    });
}

// criterion_group!(benches, compiletime_nocopy, compiletime, runtime);
criterion_group!(benches, compiletime_nocopy);
criterion_main!(benches);
