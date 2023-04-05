use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::iter::FromIterator;
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

    pub fn stack(&self, other: &Self, axis: usize) -> Option<Self> {
        if self.shape.len() != other.shape.len() {
            return None;
        }

        let mut shape = self.shape.clone();
        shape[axis] += other.shape[axis];
        let mut data = Vec::with_capacity(shape.iter().product());
        let mut offset = 0;

        for i in 0..self.len() {
            if i % self.shape[axis] == 0 {
                offset += other.shape[axis];
            }
            data.push(self.data[i]);
        }

        for i in 0..other.len() {
            let j = i + offset;
            if j % shape[axis] == 0 {
                offset += self.shape[axis];
            }
            data.push(other.data[i]);
        }

        Some(Self::new(data, shape))
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

pub struct TensorIter<'a, T>
where
    T: Copy,
{
    tensor: &'a Tensor<T>,
    index: usize,
}

impl<'a, T> Iterator for TensorIter<'a, T>
where
    T: Copy,
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
    T: Copy,
{
    pub fn iter(&self) -> TensorIter<T> {
        TensorIter {
            tensor: self,
            index: 0,
        }
    }
}

impl<T> FromIterator<T> for Tensor<T>
where
    T: Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        let shape = vec![data.len()];
        Self::new(data, shape)
    }
}
