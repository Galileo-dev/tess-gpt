use tess_math::math::Tensor;

fn main() {
    println!("========================");
    println!("=        Tensor        =");
    println!("========================");

    let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    println!("{:?}", tensor);

    // 3x3 tensor
    let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]);
    println!("{:?}", tensor);

    // stack 3 tensors
    let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
    let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
    let tensor = Tensor::stack(&vec![tensor1, tensor2, tensor3], 0);
    println!("{:?}", tensor);

    let tensor1 = Tensor::new(vec![1, 2], vec![2, 1]);

    println!("{:?}", tensor1);
    // get the first element of the tensor
    println!("{:?}", tensor1.get(0, 0));
}
