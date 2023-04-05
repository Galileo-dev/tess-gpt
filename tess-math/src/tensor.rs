use std::fmt::Debug;
use std::ops::{Bound, RangeBounds};
use std::{
    fmt::{self, Display, Formatter},
    ops::Range,
};

pub struct Tensor<T>
where
    T: Copy,
{
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "data and shape sizes mismatch"
        );
        Self { data, shape }
    }

    pub fn get(&self, indices: &[usize]) -> T {
        let index = self.get_index(indices);
        self.data[index]
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: &[R]) -> Self {
        let indices: Vec<usize> = range
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                let start = match r.start_bound() {
                    Bound::Included(&s) | Bound::Excluded(&s) => s,
                    Bound::Unbounded => 0,
                };
                let end = match r.end_bound() {
                    Bound::Included(&e) => e + 1,
                    Bound::Excluded(&e) => e,
                    Bound::Unbounded => self.shape[i],
                };
                let step = if i == 0 { 1 } else { self.shape[i - 1] };
                (start..end).step_by(step)
            })
            .collect();

        let data = indices.iter().map(|&i| self.data[i]).collect();
        let shape = range
            .iter()
            .map(|r| match (r.start_bound(), r.end_bound()) {
                (Bound::Included(&s), Bound::Included(&e)) => e - s + 1,
                (Bound::Included(&s), Bound::Excluded(&e)) => e - s,
                (Bound::Included(&s), Bound::Unbounded) => self.shape[range.len() - 1] - s,
                (Bound::Excluded(&s), Bound::Included(&e)) => e - s,
                (Bound::Excluded(&s), Bound::Excluded(&e)) => e - s - 1,
                (Bound::Excluded(&s), Bound::Unbounded) => self.shape[range.len() - 1] - s,
                (Bound::Unbounded, Bound::Included(&e)) => e + 1,
                (Bound::Unbounded, Bound::Excluded(&e)) => e,
                (Bound::Unbounded, Bound::Unbounded) => self.shape[range.len() - 1],
            })
            .collect();

        Self::new(data, shape)
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        let index = self.get_index(indices);
        self.data[index] = value;
    }

    pub fn get_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "wrong number of indices");
        let mut index = 0;
        for i in 0..indices.len() {
            assert!(indices[i] < self.shape[i], "index out of range");
            index = index * self.shape[i] + indices[i];
        }
        index
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Display for Tensor<T>
where
    T: Copy + Display + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "tensor(")?;
        for (i, dim) in self.shape.iter().enumerate() {
            write!(f, "[")?;
            for j in 0..*dim {
                let index = match i {
                    0 => j,
                    _ => j * self.shape[i - 1],
                };
                write!(f, "{}", self.data[index])?;
                if j != *dim - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i != self.shape.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

impl<T> Debug for Tensor<T>
where
    T: Copy + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor {{ shape: {:?}, data: {:?} }}",
            self.shape, self.data
        )
    }
}
