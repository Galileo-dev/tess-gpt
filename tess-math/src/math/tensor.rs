use rand::prelude::*;
use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

// ===========================================================
/// A simple tensor library.
// Author: Galileo-Dev
// Date: 2023-04-09
// ===========================================================

#[derive(Debug, PartialEq)]
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
        };

        let mut data = Vec::new();

        for row in &row_indices {
            for col in &column_indices {
                data.push(self.index(*row, *col));
            }
        }

        Tensor::new(data, vec![row_indices.len(), column_indices.len()])
    }

    fn index(&self, row: usize, col: usize) -> T {
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

        // Test getting a sub-tensor with a single row or column
        assert_eq!(tensor.get(1..2, 0..3).data(), &[4, 5, 6]);
        assert_eq!(tensor.get(0..2, 2..=2).data(), &[3, 6]);
        assert_eq!(tensor.get(1..2, &[0, 1, 2]).data(), &[4, 5, 6]);
    }
}
