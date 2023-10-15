use super::LanguageModel;
use std::fmt::Debug;
use std::fmt::Display;

pub struct BigramLanguageModel();

impl BigramLanguageModel {
    pub fn new() -> Self {
        BigramLanguageModel {}
    }
}

pub struct LanguageModelDataset<T> {
    pub data: Vec<T>,
}

impl LanguageModelDataset<String> {
    pub fn next(&self) -> Option<String> {
        unimplemented!()
    }
}

impl<T> LanguageModel<T> for BigramLanguageModel
where
    T: Debug + Display,
{
    fn train(&mut self, _dataset: &LanguageModelDataset<T>) {
        // get a pair of characters from the dataset
        unimplemented!()
    }

    fn score(&self, _dataset: &LanguageModelDataset<T>) -> f64 {
        unimplemented!()
    }
}
