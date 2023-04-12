use rand::prelude::*;
use std::{
    fmt::{Debug, Formatter},
    mem,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    slice,
};

// ===========================================================
/// A simple tensor library.
// Author: Galileo-Dev
// Date: 2023-04-09
// ===========================================================

#[derive(PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        // if shape is 1 dimensional, convert it to a 2 dimensional tensor
        // for example a [2] shape becomes a [2, 1] shape
        // and a [3] shape becomes a [3, 1] shape
        let shape = if shape.len() == 1 {
            vec![shape[0], 1]
        } else {
            shape
        };

        Tensor { shape, data }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn randint(low: i32, high: i32, shape: Vec<usize>, seed: u64) -> Tensor<i32> {
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

        Tensor::new(data, shape)
    }
}

pub enum MyRange {
    Empty,
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeTo(RangeTo<usize>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<usize>),
    RangeToInclusive(RangeToInclusive<usize>),
}

pub enum Indices {
    Vec(Vec<usize>),
    Range(MyRange),
    Single(usize),
}

// from range
impl From<Range<usize>> for Indices {
    fn from(range: Range<usize>) -> Indices {
        Indices::Range(MyRange::Range(range))
    }
}

impl From<std::ops::RangeTo<usize>> for Indices {
    fn from(range: std::ops::RangeTo<usize>) -> Indices {
        Indices::Range(MyRange::RangeTo(range))
    }
}

impl From<std::ops::RangeFrom<usize>> for Indices {
    fn from(range: std::ops::RangeFrom<usize>) -> Indices {
        Indices::Range(MyRange::RangeFrom(range))
    }
}

impl From<std::ops::RangeFull> for Indices {
    fn from(range: std::ops::RangeFull) -> Indices {
        Indices::Range(MyRange::RangeFull(range))
    }
}

impl From<std::ops::RangeInclusive<usize>> for Indices {
    fn from(range: std::ops::RangeInclusive<usize>) -> Indices {
        Indices::Range(MyRange::RangeInclusive(range))
    }
}

impl From<std::ops::RangeToInclusive<usize>> for Indices {
    fn from(range: std::ops::RangeToInclusive<usize>) -> Indices {
        Indices::Range(MyRange::RangeToInclusive(range))
    }
}

impl<const N: usize> From<&[usize; N]> for Indices {
    fn from(slice: &[usize; N]) -> Self {
        let mut indices = Vec::new();
        for i in slice {
            indices.push(*i as usize);
        }
        Indices::Vec(indices)
    }
}

impl From<usize> for Indices {
    fn from(index: usize) -> Self {
        Indices::Vec(vec![index])
    }
}

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
    /// tensor.get(1, ..2);
    ///
    /// tensor.get(.., 1);
    ///
    /// tensor.get(.., ..2);
    pub fn get<R, C>(&self, row: R, column: C) -> Self
    where
        for<'a> R: Into<Indices>,
        C: for<'a> Into<Indices>,
    {
        let row_indices: Vec<usize> = match row.into() {
            Indices::Range(range) => match range.into() {
                MyRange::Range(range) => range_to_indices(range, self.shape[0]),
                MyRange::RangeFrom(range) => range_to_indices(range, self.shape[0]),
                MyRange::RangeTo(range) => range_to_indices(range, self.shape[0]),
                MyRange::RangeFull(range) => range_to_indices(range, self.shape[0]),
                MyRange::RangeInclusive(range) => range_to_indices(range, self.shape[0]),
                MyRange::RangeToInclusive(range) => range_to_indices(range, self.shape[0]),
                MyRange::Empty => vec![],
            },

            Indices::Vec(indices) => indices,
            Indices::Single(index) => vec![index],
        };

        let column_indices: Vec<usize> = match column.into() {
            Indices::Range(range) => match range.into() {
                MyRange::Range(range) => range_to_indices(range, self.shape[1]),
                MyRange::RangeFrom(range) => range_to_indices(range, self.shape[1]),
                MyRange::RangeTo(range) => range_to_indices(range, self.shape[1]),
                MyRange::RangeFull(range) => range_to_indices(range, self.shape[1]),
                MyRange::RangeInclusive(range) => range_to_indices(range, self.shape[1]),
                MyRange::RangeToInclusive(range) => range_to_indices(range, self.shape[1]),
                MyRange::Empty => vec![],
            },
            Indices::Vec(indices) => indices,
            Indices::Single(index) => vec![index],
        };

        let mut data = Vec::new();

        for row in &row_indices {
            for col in &column_indices {
                data.push(self.index(*row, *col));
            }
        }

        Tensor::new(data, vec![row_indices.len(), column_indices.len()])
    }

    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn transpose(&mut self, axis1: usize, axis2: usize) {
        let n = self.shape.len();
        assert!(axis1 < n && axis2 < n);

        let (a, b) = if axis1 < axis2 {
            (axis1, axis2)
        } else {
            (axis2, axis1)
        };

        let mut indices = (0..n).collect::<Vec<_>>();
        indices.swap(a, b);

        let mut new_data = vec![mem::MaybeUninit::<T>::uninit(); self.num_elements()];
        let mut strides = vec![1; n];
        for i in (0..n).rev() {
            strides[indices[i]] = if i == n - 1 {
                1
            } else {
                strides[indices[i + 1]] * self.shape[indices[i + 1]]
            };
        }

        for (i, element) in self.data.iter().enumerate() {
            let mut index = i;
            let mut new_index = 0;
            for j in 0..n {
                let stride = strides[indices[j]];
                let dim_size = self.shape[indices[j]];
                let old_index = index / stride % dim_size;
                new_index += old_index * stride;
                index -= old_index * stride;
            }
            new_data[new_index] = mem::MaybeUninit::new(element.clone());
        }

        self.data = unsafe { new_data.into_iter().map(|x| x.assume_init()).collect() };
        self.shape = indices.into_iter().map(|i| self.shape[i]).collect();
    }

    /// stacks multiple tensors together given an axis
    ///
    /// # Examples
    /// ```
    /// use tess_math::math::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// let tensor1 = &[1.0, 2.0, 3.0];
    /// let tensor2 = &[4.0, 5.0, 6.0, 7.0];
    /// let tensor3 = &[8.0];
    ///
    /// let stacked = stack_tensors(&[tensor1, tensor2, tensor3], 0);
    /// assert_eq!(stacked, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    pub fn stack(tensors: &[Tensor<T>], axis: usize) -> Tensor<T> {
        let mut new_shape = tensors[0].shape.clone();
        new_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

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
            assert_eq!(tensor.shape.len(), new_shape.len());
            for (i, dim_size) in tensor.shape.iter().enumerate() {
                if i != axis {
                    assert_eq!(dim_size, &new_shape[i]);
                }
            }
            for (i, element) in tensor.data.iter().enumerate() {
                let mut index = i;
                let mut new_index = offset;
                for j in 0..new_shape.len() {
                    let stride = strides[j];
                    let dim_size = new_shape[j];
                    let old_index = index / stride % dim_size;
                    new_index += old_index * stride;
                    index -= old_index * stride;
                }
                new_data.push(element.clone());
            }
            offset += tensor.shape[axis];
        }
        Tensor::new(new_data, new_shape)
    }

    fn index(&self, row: usize, col: usize) -> T {
        // check if the indices are in bounds
        assert!(row < self.shape[0] && col < self.shape[1]);
        self.data[row * self.shape[1] + col]
    }
}

fn range_to_indices(range: impl std::ops::RangeBounds<usize>, size: usize) -> Vec<usize> {
    let start = match range.start_bound() {
        std::ops::Bound::Included(&n) => n,
        std::ops::Bound::Excluded(&n) => n + 1,
        std::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        std::ops::Bound::Included(&n) => n + 1,
        std::ops::Bound::Excluded(&n) => n,
        std::ops::Bound::Unbounded => size,
    };
    (start..end).collect()
}

// Iterator over the indices of a tensor
pub struct TensorIter<'a, T>
where
    T: Copy + Debug,
{
    tensor: &'a Tensor<T>,
    index: usize,
}

impl<'a, T> Iterator for TensorIter<'a, T>
where
    T: Copy + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.tensor.len() {
            let result = Some(&self.tensor.data[self.index]);
            self.index += 1;
            result
        } else {
            None
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    pub fn iter(&self) -> TensorIter<T> {
        TensorIter {
            tensor: self,
            index: 0,
        }
    }
}

impl<T> FromIterator<Tensor<T>> for Tensor<T>
where
    T: Copy + Debug,
{
    fn from_iter<I: IntoIterator<Item = Tensor<T>>>(iter: I) -> Self {
        let mut tensors = iter.into_iter();
        let mut data = Vec::new();
        let mut shape = Vec::new();
        let mut output: Vec<Tensor<T>> = Vec::new();
        if let Some(first_tensor) = tensors.next() {
            shape.push(1);
            shape.extend(first_tensor.shape.clone());
            data.extend(first_tensor.data.clone());
            output.push(first_tensor);
        }
        for tensor in tensors {
            shape[0] += 1;
            data.extend(tensor.data.clone());
            output.push(tensor);
        }
        Self::new(data, shape)
    }
}

// impl Debug
impl<T> Debug for Tensor<T>
where
    T: Copy + Debug + PartialEq<T>,
{
    // add new line after each row in the tensor
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // pytorch style
        let mut s = String::new();

        let indent1 = " ".repeat(8);
        let indent2 = " ".repeat(15);

        s.push_str("Tensor{ shape: ");
        s.push_str(&format!("{:?}", self.shape));
        s.push_str(",\n");
        s.push_str(&indent1);
        s.push_str("data:\n");

        // add to the start of each row from this point on

        let mut index = 0;
        for i in 0..self.shape[0] {
            s.push_str(&indent2);
            s.push_str("[");
            for j in 0..self.shape[1] {
                s.push_str(&format!("{:?}", self.index(i, j)));
                if j != self.shape[1] - 1 {
                    s.push_str(", ");
                }
            }
            s.push_str("]");
            s.push_str(",");
            if i != self.shape[0] - 1 {
                s.push_str("\n");
            }
        }
        s.push_str("\n}");
        write!(f, "{}", s);

        Ok(())
    }
}

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
    #[should_panic]
    fn test_new_panic() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 4]; // incorrect shape
        let _tensor = Tensor::new(data, shape);
    }

    #[test]
    fn test_get() {
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        // Test getting a single element
        assert_eq!(tensor.get(0, 0).data(), &[1]);

        // Test getting a sub-tensor with a single element
        assert_eq!(tensor.get(0..1, 0..1).data(), &[1]);

        // Test getting a sub-tensor with multiple elements in a single row
        assert_eq!(tensor.get(0..1, 0..2).data(), &[1, 2]);

        // Test getting a sub-tensor with multiple elements in a single column
        assert_eq!(tensor.get(0..2, 0..1).data(), &[1, 4]);

        // Test getting a sub-tensor with multiple elements in multiple rows and columns
        assert_eq!(tensor.get(0..2, 0..2).data(), &[1, 2, 4, 5]);

        // Test getting the entire tensor
        assert_eq!(tensor.get(0..2, 0..3).data(), &[1, 2, 3, 4, 5, 6]);

        assert_eq!(tensor.get(1, 2).data(), &[6]);

        assert_eq!(tensor.get(.., 1).data(), &[2, 5]);

        // Test getting a sub-tensor with a single row or column
        assert_eq!(tensor.get(1..2, 0..3).data(), &[4, 5, 6]);
        assert_eq!(tensor.get(0..2, 2..=2).data(), &[3, 6]);
        assert_eq!(tensor.get(1..2, &[0, 1, 2]).data(), &[4, 5, 6]);
    }

    #[test]
    fn test_stack() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
        let tensor = Tensor::stack(&vec![tensor1, tensor2, tensor3], 0);

        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(tensor.shape(), &[6, 2]);

        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let tensor3 = Tensor::new(vec![9, 10, 11, 12], vec![2, 2]);
        let tensor = Tensor::stack(&vec![tensor1, tensor2, tensor3], 1);

        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(tensor.shape(), &[2, 6]);
    }
}
