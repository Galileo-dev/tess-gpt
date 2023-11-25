use rand::prelude::*;
use rayon::range;
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
// Desciption: A simple tensor library that supports 1D and 2D tensors.
// ===========================================================

#[derive(PartialEq, Eq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

/// Simple tensor
impl<T> Tensor<T>
where
    T: Copy + Debug,
{
    #[must_use]
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());

        let shape = if shape.len() == 1 {
            vec![shape[0], 1]
        } else {
            shape
        };

        Self { shape, data }
    }

    #[must_use]
    pub const fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    #[must_use]
    pub const fn data(&self) -> &Vec<T> {
        &self.data
    }

    #[must_use]
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Default,
    {
        let data = vec![T::default(); shape.iter().product()];
        Self::new(data, shape)
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

/// Implementing Debug for Tensor
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
