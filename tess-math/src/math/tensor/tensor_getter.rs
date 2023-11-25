use super::tensor::Tensor;
use std::fmt::Debug;

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    /// Returns a tensor from the given row and column indices of the original tensor.
    ///
    /// # Examples
    /// ```
    /// use tess_math::math::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    ///
    /// tensor.get(0, 1);
    ///
    /// tensor.get(1, ..2, 1..=5);
    ///
    /// tensor.get(1.., 2.., 5);
    ///
    /// tensor.get(.., ..2);
    #[must_use]
    pub fn get(self, indices: &[usize]) -> Self {
        return self;
    }

    /// Returns a tensor from the given row and column indices of the original tensor.
    /// This is a helper function for the get method.
    /// # Examples
    /// ```
    /// use tess_math::math::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    ///
    /// tensor.get_indices(&[0, 1], 0);
    ///
    /// tensor.get_indices(&[0, 1], 1);
    ///
    /// tensor.get_indices(&[0, 1], 2);
    ///
    /// tensor.get_indices(&[0, 1], 3);
    ///
    /// tensor.get_indices(&[0, 1], 4);
    ///
    #[must_use]
    pub fn get_indices(self, indices: &[usize], axis: usize) -> Self {
        let mut new_shape = self.shape().clone();
        new_shape.remove(axis);

        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        let mut strides = vec![1; self.shape().len()];
        for i in (0..self.shape().len()).rev() {
            strides[i] = if i == self.shape().len() - 1 {
                1
            } else {
                strides[i + 1] * self.shape()[i + 1]
            };
        }

        for index in indices {
            assert!(index < &self.shape()[axis]);
            let mut offset = 0;
            for i in 0..self.shape().len() {
                if i != axis {
                    offset += index * strides[i];
                }
            }
            for i in 0..self.shape()[axis] {
                let index = offset + i * strides[axis];
                new_data.push(self.data()[index]);
            }
        }

        Self::new(new_data, new_shape)
    }
}
