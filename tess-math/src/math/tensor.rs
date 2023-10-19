use rand::prelude::*;
use std::fmt::Display;
use std::{
    fmt::{self, Debug, Formatter},
    mem,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

// ===========================================================
// A simple tensor library.
// Author: Galileo-Dev
// Date: 2023-04-09
// ===========================================================

#[derive(PartialEq, Eq)]
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

        Self { shape, data }
    }

    pub const fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub const fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Default,
    {
        let data = vec![T::default(); shape.iter().product()];
        Self::new(data, shape)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Debug + PartialOrd + rand::distributions::uniform::SampleUniform,
{
    pub fn randn(low: T, high: T, shape: Vec<usize>, seed: u64) -> Tensor<T> {
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
    fn from(range: Range<usize>) -> Self {
        Self::Range(MyRange::Range(range))
    }
}

impl From<std::ops::RangeTo<usize>> for Indices {
    fn from(range: std::ops::RangeTo<usize>) -> Self {
        Self::Range(MyRange::RangeTo(range))
    }
}

impl From<std::ops::RangeFrom<usize>> for Indices {
    fn from(range: std::ops::RangeFrom<usize>) -> Self {
        Self::Range(MyRange::RangeFrom(range))
    }
}

impl From<std::ops::RangeFull> for Indices {
    fn from(range: std::ops::RangeFull) -> Self {
        Self::Range(MyRange::RangeFull(range))
    }
}

impl From<std::ops::RangeInclusive<usize>> for Indices {
    fn from(range: std::ops::RangeInclusive<usize>) -> Self {
        Self::Range(MyRange::RangeInclusive(range))
    }
}

impl From<std::ops::RangeToInclusive<usize>> for Indices {
    fn from(range: std::ops::RangeToInclusive<usize>) -> Self {
        Self::Range(MyRange::RangeToInclusive(range))
    }
}

impl<const N: usize> From<&[usize; N]> for Indices {
    fn from(slice: &[usize; N]) -> Self {
        let mut indices = Vec::new();
        for i in slice {
            indices.push(*i);
        }
        Self::Vec(indices)
    }
}

impl From<usize> for Indices {
    fn from(index: usize) -> Self {
        Self::Vec(vec![index])
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
    #[must_use]
    pub fn get<R, C>(&self, row: R, column: C) -> Self
    where
        R: for<'a> Into<Indices>,
        C: for<'a> Into<Indices>,
    {
        let row_indices: Vec<usize> = match row.into() {
            Indices::Range(range) => match range {
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
            Indices::Range(range) => match range {
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

        Self::new(data, vec![row_indices.len(), column_indices.len()])
    }

    pub fn set<R, C, V>(&mut self, row: R, column: C, values: V)
    where
        R: for<'a> Into<Indices>,
        C: for<'a> Into<Indices>,
        V: IntoIterator<Item = T>,
    {
        let row_indices: Vec<usize> = Self::map_indices(row, self.shape[0]);
        let column_indices: Vec<usize> = Self::map_indices(column, self.shape[1]);

        let mut value_iter = values.into_iter();

        for row in &row_indices {
            for col in &column_indices {
                let value = value_iter.next().expect("Not enough values provided");
                *self.index_mut(*row, *col) = value;
            }
        }

        assert!(value_iter.next().is_none(), "Too many values provided");
    }

    fn map_indices<I: for<'a> Into<Indices>>(indices: I, shape_dim: usize) -> Vec<usize> {
        match indices.into() {
            Indices::Range(range) => match range {
                MyRange::Range(range) => range_to_indices(range, shape_dim),
                MyRange::RangeFrom(range) => range_to_indices(range, shape_dim),
                MyRange::RangeTo(range) => range_to_indices(range, shape_dim),
                MyRange::RangeFull(range) => range_to_indices(range, shape_dim),
                MyRange::RangeInclusive(range) => range_to_indices(range, shape_dim),
                MyRange::RangeToInclusive(range) => range_to_indices(range, shape_dim),
                MyRange::Empty => vec![],
            },
            Indices::Vec(indices) => indices,
            Indices::Single(index) => vec![index],
        }
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
            new_data[new_index] = mem::MaybeUninit::new(*element);
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
                new_data.push(*element);
            }
            offset += tensor.shape[axis];
        }
        Self::new(new_data, new_shape)
    }

    fn index(&self, row: usize, col: usize) -> T {
        // check if the indices are in bounds
        assert!(row < self.shape[0] && col < self.shape[1]);
        self.data[row * self.shape[1] + col]
    }

    pub fn index_mut(&mut self, row: usize, col: usize) -> &mut T {
        // Check if the indices are in bounds
        assert!(row < self.shape[0] && col < self.shape[1]);
        let index = row * self.shape[1] + col;
        &mut self.data[index]
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

// Implementing Iterator for Tensor
pub struct TensorIter<'a, T> {
    index: usize,
    tensor: &'a Tensor<T>,
}

impl<'a, T> Iterator for TensorIter<'a, T>
where
    T: Copy + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.tensor.len() {
            return None;
        }
        let result = &self.tensor.data[self.index];
        self.index += 1;
        Some(result)
    }
}

// Adding the method to get the iterator from Tensor
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

impl<T> FromIterator<Tensor<T>> for Tensor<T>
where
    T: Copy + Debug,
{
    fn from_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        let tensor_vec: Vec<Self> = iter.into_iter().collect();

        // Error handling: return an empty tensor if no items
        if tensor_vec.is_empty() {
            return Self::new(Vec::new(), Vec::new());
        }

        // Call your existing stack function here
        // Assuming the function signature is something like:
        // fn stack_tensors(tensors: &[Tensor<T>]) -> Tensor<T>

        Self::stack(&tensor_vec, 0)
    }
}

impl<T: Display> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let indent1 = " ".repeat(8);
        let indent2 = " ".repeat(15);

        write!(f, "Tensor{{ shape: {:?},\n{}data: [", self.shape, indent1)?;

        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            for row in 0..rows {
                if row != 0 {
                    write!(f, ",\n{indent2}")?;
                }

                let mut row_str = String::new();
                for col in 0..cols {
                    if col != 0 {
                        row_str.push_str(", ");
                    }
                    let index = row * cols + col;
                    row_str.push_str(&self.data[index].to_string());
                }
                write!(f, "[{row_str}]")?;
            }
        }

        write!(f, "] }}")
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
    #[should_panic(expected = "assertion `left == right` failed\n  left: 6\n right: 8")]
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

        // Test a 1D tensor
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![6]);
        assert_eq!(tensor.get(0..=1, ..).data(), &[1, 2]);
        assert_eq!(tensor.get(0..=3, ..).data(), &[1, 2, 3, 4]);
        assert_eq!(tensor.get(0..=5, ..).data(), &[1, 2, 3, 4, 5, 6]);

        // Tensor 1x100
        let tensor = Tensor::new((0..100).collect::<Vec<_>>(), vec![1, 100]);
        // get the first 50 elements
        assert_eq!(tensor.get(0..1, 0..50).data(), &(0..50).collect::<Vec<_>>());
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
    fn test_iter_map() {
        // 2x3 tensor
        let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let tensor2 = tensor.iter().map(|x| x * 2).collect::<Vec<_>>();
        assert_eq!(tensor2, vec![2, 4, 6, 8, 10, 12]);
    }
}
