// ===========================================================
/// A simple tensor library.
// Author: Galileo-Dev
// Date: 2023-04-09
// ===========================================================
use std::{
    fmt::Debug,
    ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use rand::prelude::*;

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

    pub fn stack(tensors: &[&Self], axis: usize) -> Option<Self> {
        if tensors.is_empty() {
            return None;
        }
        let shape: Vec<usize> = {
            let mut shape = tensors[0].shape.clone();
            shape.insert(axis, tensors.len());
            shape
        };
        let mut data: Vec<T> = Vec::with_capacity(shape.iter().product());
        for i in 0..shape[axis] {
            for tensor in tensors {
                let mut indices: Vec<usize> = vec![i];
                for j in 0..tensor.shape.len() {
                    if j == axis {
                        indices.extend(0..1);
                    } else {
                        indices.push(0);
                    }
                }
                for j in 0..tensor.shape[axis] {
                    indices[axis + 1] = j;
                    let value = tensor.get(.., indices);
                    data.push(value);
                }
            }
        }
        Some(Self::new(data, shape))
    }
}

pub enum Index {
    Index(usize),
    Range(Bound<usize>, Bound<usize>),
}

impl From<usize> for Index {
    fn from(idx: usize) -> Self {
        Index::Index(idx)
    }
}

impl From<RangeFull> for Index {
    fn from(_: RangeFull) -> Self {
        Index::Range(Bound::Unbounded, Bound::Unbounded)
    }
}

impl From<Range<usize>> for Index {
    fn from(range: Range<usize>) -> Self {
        Index::Range(Bound::Included(range.start), Bound::Excluded(range.end))
    }
}

impl From<RangeTo<usize>> for Index {
    fn from(range: RangeTo<usize>) -> Self {
        Index::Range(Bound::Unbounded, Bound::Excluded(range.end))
    }
}

impl From<RangeInclusive<usize>> for Index {
    fn from(range: RangeInclusive<usize>) -> Self {
        Index::Range(
            Bound::Included(*range.start()),
            Bound::Included(*range.end()),
        )
    }
}

impl From<RangeToInclusive<usize>> for Index {
    fn from(range: RangeToInclusive<usize>) -> Self {
        Index::Range(Bound::Unbounded, Bound::Included(range.end))
    }
}

impl From<RangeFrom<usize>> for Index {
    fn from(range: RangeFrom<usize>) -> Self {
        Index::Range(Bound::Included(range.start), Bound::Unbounded)
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
        R: Into<Indices>,
        C: Into<Indices>,
    {
        let row_ranges = match row.into() {
            Index::Index(row) => (row..(row + 1)),
            Index::Range(start, end) => {
                let start = match start {
                    Bound::Included(start) => start,
                    Bound::Excluded(start) => start + 1,
                    Bound::Unbounded => 0,
                };

                let end = match end {
                    Bound::Included(end) => end + 1,
                    Bound::Excluded(end) => end,
                    Bound::Unbounded => self.shape[0],
                };

                (start..end)
            }
        };

        let col_ranges = match column.into() {
            Index::Index(col) => (col..(col + 1)),
            Index::Range(start, end) => {
                let start = match start {
                    Bound::Included(start) => start,
                    Bound::Excluded(start) => start + 1,
                    Bound::Unbounded => 0,
                };

                let end = match end {
                    Bound::Included(end) => end + 1,
                    Bound::Excluded(end) => end,
                    Bound::Unbounded => self.shape[1],
                };

                (start..end)
            }
        };

        let indices = row_ranges
            .clone()
            .flat_map(|row| col_ranges.clone().map(move |col| self.index(row, col)))
            .collect();

        let shape = vec![row_ranges.len(), col_ranges.len()];

        Self::new(indices, shape)
    }

    fn index(&self, row: usize, col: usize) -> T {
        self.data[row * self.shape[1] + col]
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Slice {
    Index(isize),
    Slice(isize, isize, isize),
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
        assert_eq!(tensor.get(0..2, 2..3).data(), &[3, 6]);
    }
}
