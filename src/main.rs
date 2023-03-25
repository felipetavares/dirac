use dirac::xdirac as dirac;
use tensor::ToTensor;

fn main() {
    dbg!(dirac!((|0> + |1>) / ||0> + |1>|));
    dbg!(dirac!(|+>));
}
