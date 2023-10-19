use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::fmt::{self, Display, Formatter};
use std::iter::FromIterator;
use std::ops::{Bound, Range, RangeBounds, RangeTo};

#[derive(Clone)]
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
                // Handle range bounds for the selected dimension (assuming i == 0)
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

                // Handle range bounds for other dimensions
                let start = if i == 0 { start } else { 0 };
                let end = if i == 0 { end } else { self.shape[i] };
                let step = if i == 0 { step } else { 1 };

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

    pub fn get_slice(&self, b: usize, r: Range<usize>) -> &[T] {
        let slice_start = b * self.shape[1] + r.start;
        let slice_end = b * self.shape[1] + r.end;
        &self.data[slice_start..slice_end]
    }

    pub fn get_element(&self, b: usize, t: usize) -> T {
        let index = b * self.shape[1] + t;
        self.data[index]
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

    pub fn slice(&self, start: usize, end: usize, step: usize) -> Self {
        let mut data = Vec::new();
        for i in (start..end).step_by(step) {
            data.push(self.data[i]);
        }
        let mut shape = self.shape.clone();
        shape[0] = data.len();
        Self::new(data, shape)
    }

    pub fn randn(low: i32, high: i32, shape: Vec<usize>, seed: u64) -> Tensor<i32> {
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

    pub fn len(&self) -> usize {
        self.data.len()
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
                    let value = tensor.get(&indices);
                    data.push(value);
                }
            }
        }
        Some(Self::new(data, shape))
    }
}

impl<T> Display for Tensor<T>
where
    T: Copy + Display + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {{ shape: {:?},\n", self.shape)?;
        write!(f, "data: \n[")?;
        let mut i = 0;
        for val in self.data.iter() {
            write!(f, "{}", val)?;
            i += 1;
            if i < self.data.len() {
                write!(f, ", ")?;
            }
            if i % self.shape[self.shape.len() - 1] == 0 {
                write!(f, "\n ")?;
            }
        }
        write!(f, "] \n}}")
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
