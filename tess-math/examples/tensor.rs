use tess_math::math::Tensor;

fn main() {
    let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    println!("{:?}", tensor);

    println!("{:?}", tensor.get(0, 1));

    println!("{:?}", tensor.get(1, ..2));

    println!("{:?}", tensor.get(.., 1));

    println!("{:?}", tensor.get(.., ..2));

    println!("{:?}", tensor.get(.., ..));

    println!("{:?}", tensor.get(.., 1..));
}
