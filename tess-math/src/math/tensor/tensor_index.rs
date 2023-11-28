pub trait TensorIndex<T> {
    // Define methods or associated types if needed
    fn get_indices(&self, tdim_size: usize) -> Vec<usize>;
}

// Implement the trait for relevant types
impl<T> TensorIndex<T> for usize {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        vec![*self]
    }
}
impl<T> TensorIndex<T> for std::ops::Range<usize> {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        self.clone().collect()
    }
}
impl<T> TensorIndex<T> for std::ops::RangeInclusive<usize> {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        self.clone().collect()
    }
}
impl<T> TensorIndex<T> for std::ops::RangeFrom<usize> {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        self.clone().collect()
    }
}
impl<T> TensorIndex<T> for std::ops::RangeTo<usize> {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        (0..self.end).collect()
    }
}
impl<T> TensorIndex<T> for std::ops::RangeToInclusive<usize> {
    fn get_indices(&self, _dim_size: usize) -> Vec<usize> {
        (0..=self.end).collect()
    }
}

impl<T> TensorIndex<T> for std::ops::RangeFull {
    fn get_indices(&self, dim_size: usize) -> Vec<usize> {
        (0..dim_size).collect()
    }
}
