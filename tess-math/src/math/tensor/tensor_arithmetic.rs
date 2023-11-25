use super::tensor::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fmt::Debug;

impl<T> Tensor<T>
where
    T: Copy + Debug + PartialOrd + rand::distributions::uniform::SampleUniform,
{
    /// # Panics
    /// Panics if low is greater than high
    pub fn randn(low: T, high: T, shape: Vec<usize>, seed: u64) -> Self {
        assert!(low <= high, "low must be less than or equal to high");
        // if no seed is provided, use the current time as a seed
        let mut rng = if seed == 0 {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(seed)
        };

        let mut data = Vec::new();
        for _ in 0..shape.iter().product() {
            let value = rng.gen_range(low..=high);
            data.push(value);
        }

        Self::new(data, shape)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    /// stacks multiple tensors together given an axis
    ///
    /// # Examples
    /// ```
    /// use tess_math::math::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
    /// let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
    /// let tensor = Tensor::stack(&[tensor1, tensor2, tensor3], 0);
    ///
    /// assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    /// assert_eq!(tensor.shape(), &[6, 2]);
    /// ```
    #[must_use]
    pub fn stack(tensors: &[Self], axis: usize) -> Self {
        let mut new_shape = tensors[0].shape().clone();
        new_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        let mut strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len()).rev() {
            strides[i] = if i == new_shape.len() - 1 {
                1
            } else {
                strides[i + 1] * new_shape[i + 1]
            };
        }

        let mut offset = 0;
        for tensor in tensors {
            assert_eq!(tensor.shape().len(), new_shape.len());
            for (i, dim_size) in tensor.shape().iter().enumerate() {
                if i != axis {
                    assert_eq!(dim_size, &new_shape[i]);
                }
            }
            for (i, element) in tensor.data().iter().enumerate() {
                let mut index = i;
                let mut new_index = offset;
                for j in 0..new_shape.len() {
                    let stride = strides[j];
                    let dim_size = new_shape[j];
                    let old_index = index / stride % dim_size;
                    new_index += old_index * stride;
                    index -= old_index * stride;
                }
                new_data.push(*element);
            }
            offset += tensor.shape()[axis];
        }
        Self::new(new_data, new_shape)
    }

    fn index(&self, row: usize, col: usize) -> T {
        // check if the indices are in bounds
        assert!(row < self.shape()[0] && col < self.shape()[1]);
        self.data()[row * self.shape()[1] + col]
    }

    pub fn index_mut(&mut self, row: usize, col: usize) -> &mut T {
        // Check if the indices are in bounds
        assert!(row < self.shape()[0] && col < self.shape()[1]);
        let index = row * self.shape()[1] + col;
        &mut self.data_mut()[index]
    }
}
