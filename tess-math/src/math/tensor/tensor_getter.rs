use super::{tensor::Tensor, tensor_index::TensorIndex};
use std::fmt::Debug;

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    // For a single index
    #[must_use]
    pub fn get_1d<I>(&self, index: &I) -> Self
    where
        I: TensorIndex<T>,
    {
        let indices = index.get_indices(self.shape()[0]);
        let result_tensor = self.get(&indices, 0);
        result_tensor
    }

    #[must_use]
    pub fn get_2d<I1, I2>(&self, index: &(I1, I2)) -> Self
    where
        I1: TensorIndex<T>,
        I2: TensorIndex<T>,
    {
        let indices_1 = index.0.get_indices(self.shape()[0]);
        let result_tensor_1 = self.get(&indices_1, 0);
        let indices_2 = index.1.get_indices(self.shape()[1]);
        result_tensor_1.get(&indices_2, 1)
    }

    #[must_use]
    pub fn get_3d<I1, I2, I3>(&self, index: &(I1, I2, I3)) -> Self
    where
        I1: TensorIndex<T>,
        I2: TensorIndex<T>,
        I3: TensorIndex<T>,
    {
        let indices_1 = index.0.get_indices(self.shape()[0]);
        let result_tensor_1 = self.get(&indices_1, 0);
        let indices_2 = index.1.get_indices(self.shape()[1]);
        let result_tensor_2 = result_tensor_1.get(&indices_2, 1);
        let indices_3 = index.1.get_indices(self.shape()[2]);
        result_tensor_2.get(&indices_3, 2)
    }

    #[must_use]

    /// Returns a tensor from the given indices and dimension of the original tensor.
    /// # Panics
    /// Panics if any of the indices are out of bounds.
    pub fn get(&self, indices: &[usize], dim: usize) -> Self
    where
        T: Clone,
    {
        // Check if indices are out of bounds
        assert!(
            !indices.iter().any(|&index| index >= self.shape()[dim]),
            "Index out of bounds"
        );

        // Calculate total stride for dimensions before 'dim'
        let pre_stride: usize = self.shape()[..dim].iter().product();

        // Calculate stride for 'dim' and subsequent dimensions
        let post_stride: usize = self.shape()[dim + 1..].iter().product();

        // Calculate the total number of elements to be extracted
        let mut new_data = Vec::with_capacity(indices.len() * post_stride);

        for &index in indices {
            let start = index * pre_stride * post_stride;
            let end = start + post_stride;
            for i in start..end {
                new_data.push(self.data()[i]);
            }
        }

        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&self.shape()[..dim]);
        new_shape.push(indices.len());
        new_shape.extend_from_slice(&self.shape()[dim + 1..]);

        Self::new(new_data, new_shape)
    }
    // ...and so on for more dimensions if needed
}

// impl<T> Tensor<T>
// where
//     T: Copy + Debug,
// {
//     /// Returns a tensor from the given row and column indices of the original tensor.
//     /// This is a helper function for the get method.
//     /// # Examples
//     /// ```
//     /// use tess_math::math::Tensor;
//     ///
//     /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
//     ///
//     /// tensor.get_indices(&[0, 1], 0);
//     ///
//     /// tensor.get_indices(&[0, 1], 1);
//     ///
//     /// tensor.get_indices(&[0, 1], 2);
//     ///
//     #[must_use]
//     pub fn get_indices(self, indices: &[usize], axis: usize) -> Self {
//         let mut new_shape = self.shape().clone();
//         new_shape.remove(axis);

//         let mut new_data = Vec::with_capacity(new_shape.iter().product());
//         let mut strides = vec![1; self.shape().len()];
//         for i in (0..self.shape().len()).rev() {
//             strides[i] = if i == self.shape().len() - 1 {
//                 1
//             } else {
//                 strides[i + 1] * self.shape()[i + 1]
//             };
//         }

//         for index in indices {
//             assert!(index < &self.shape()[axis]);
//             let mut offset = 0;
//             for i in 0..self.shape().len() {
//                 if i != axis {
//                     offset += index * strides[i];
//                 }
//             }
//             for i in 0..self.shape()[axis] {
//                 let index = offset + i * strides[axis];
//                 new_data.push(self.data()[index]);
//             }
//         }

//         Self::new(new_data, new_shape)
//     }
// }
