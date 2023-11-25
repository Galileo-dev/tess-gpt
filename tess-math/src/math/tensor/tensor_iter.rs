use super::tensor::Tensor;
use std::fmt::Debug;

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    #[must_use]
    pub fn iter(&self) -> TensorIter<T> {
        TensorIter {
            index: 0,
            tensor: self,
        }
    }
}

/// Implementing Iterator for Tensor
pub struct TensorIter<'a, T> {
    index: usize,
    tensor: &'a Tensor<T>,
}

/// Implementing Iterator for Tensor
impl<'a, T> Iterator for TensorIter<'a, T>
where
    T: Copy + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.tensor.len() {
            return None;
        }
        let result = &self.tensor.data()[self.index];
        self.index += 1;
        Some(result)
    }
}

/// Implementing `IntoIterator` for Tensor
impl<T> FromIterator<Self> for Tensor<T>
where
    T: Copy + Debug,
{
    fn from_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        let tensor_vec: Vec<Self> = iter.into_iter().collect();
        if tensor_vec.is_empty() {
            return Self::new(Vec::new(), Vec::new());
        }

        Self::stack(&tensor_vec, 0)
    }
}
