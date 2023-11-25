use super::tensor::Tensor;
use std::fmt::Debug;
impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    pub fn set(&mut self) {}
}
