pub mod tensor;
pub use tensor::Tensor;

pub mod tensor_arithmetic;

pub mod tensor_getter;
pub mod tensor_index;
pub mod tensor_iter;
pub mod tensor_setter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 6\n right: 8")]
    fn test_new_panic() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 4]; // incorrect shape
        let _tensor = Tensor::new(data, shape);
    }

    #[test]
    fn test_print() {
        // 2x3 tensor
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        // test against string (take into account the indentation)
        println!("{tensor:?}");

        let s = "Tensor{ shape: [2, 3],
        data: [[1, 2, 3],
               [4, 5, 6]] }";
        assert_eq!(format!("{tensor:?}"), s);
    }

    #[test]
    fn test_get() {
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(tensor.get(&[0], 0), Tensor::new(vec![1, 2, 3], vec![1, 3]));

        assert_eq!(
            tensor.get(&[0, 1], 0),
            Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3])
        );

        let tensor2 = Tensor::new((0..27).collect::<Vec<_>>(), vec![3, 3, 3]);
        println!("{tensor2:?}");
        assert_eq!(
            tensor2.get(&[0], 0),
            Tensor::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![1, 3, 3])
        );

        // assert_eq!(tensor.get(&[0, 1], 0), &[1, 2, 3, 4, 5, 6]);
        // Test getting a single element
        // tensor.get((Indices::Single(0),));
        // assert_eq!(tensor.get((Indices::Single(0),)).data(), &[1]);

        // Test getting a sub-tensor with a single element
        // assert_eq!(tensor.get(0..1, 0..1).data(), &[1]);

        // // Test getting a sub-tensor with multiple elements in a single row
        // assert_eq!(tensor.get(0..1, 0..2).data(), &[1, 2]);

        // // Test getting a sub-tensor with multiple elements in a single column
        // assert_eq!(tensor.get(0..2, 0..1).data(), &[1, 4]);

        // // Test getting a sub-tensor with multiple elements in multiple rows and columns
        // assert_eq!(tensor.get(0..2, 0..2).data(), &[1, 2, 4, 5]);

        // // Test getting the entire tensor
        // assert_eq!(tensor.get(0..2, 0..3).data(), &[1, 2, 3, 4, 5, 6]);

        // assert_eq!(tensor.get(1, 2).data(), &[6]);

        // assert_eq!(tensor.get(.., 1).data(), &[2, 5]);

        // // Test getting a sub-tensor with a single row or column
        // assert_eq!(tensor.get(1..2, 0..3).data(), &[4, 5, 6]);
        // assert_eq!(tensor.get(0..2, 2..=2).data(), &[3, 6]);
        // assert_eq!(tensor.get(1..2, &[0, 1, 2]).data(), &[4, 5, 6]);

        // // Test a 1D tensor
        // let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![6]);
        // assert_eq!(tensor.get(0..=1, ..).data(), &[1, 2]);
        // assert_eq!(tensor.get(0..=3, ..).data(), &[1, 2, 3, 4]);
        // assert_eq!(tensor.get(0..=5, ..).data(), &[1, 2, 3, 4, 5, 6]);

        // // Tensor 1x100
        // let tensor = Tensor::new((0..100).collect::<Vec<_>>(), vec![1, 100]);
        // // get the first 50 elements
        // assert_eq!(tensor.get(0..1, 0..50).data(), &(0..50).collect::<Vec<_>>());
    }

    #[test]
    fn test_get_1d() {
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(
            tensor.get_1d(&(0..1)),
            Tensor::new(vec![1, 2, 3], vec![1, 3])
        );
    }

    #[test]
    fn test_get_indices() {
        let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        println!("{tensor:?}");
        // get 1d tensor from 2d tensor
        // assert_eq!(
        //     tensor.get_indices(&[0, 1], 0),
        //     Tensor::new(vec![1, 2, 3, 4], vec![2, 2])
        // );
    }
    #[test]
    fn test_tensor_set() {
        // use a tensor to set another tensor, increasing it's dimensionality
        // 2x3 tensor
        let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let tensor2: Tensor<i32> = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        // set 1,2 of tensor to tensor2
        // tensor.set_from_tensor(1, 2, tensor2);
    }

    #[test]
    fn test_iter_map() {
        // 2x3 tensor
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let tensor2 = tensor.iter().map(|x| x * 2).collect::<Vec<_>>();
        assert_eq!(tensor2, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn test_stack() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
        let tensor = Tensor::stack(&[tensor1, tensor2, tensor3], 0);

        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(tensor.shape(), &[6, 2]);

        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
        let tensor = Tensor::stack(&[tensor1, tensor2, tensor3], 1);

        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(tensor.shape(), &[2, 6]);
    }
}
