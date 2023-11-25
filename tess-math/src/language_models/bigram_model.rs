use crate::math::Tensor;

use super::LanguageModel;
use super::TokenEmbeddingTable;
use std::fmt::Debug;
use std::fmt::Display;

pub struct BigramLanguageModel {
    token_embedding_table: TokenEmbeddingTable,
}

impl BigramLanguageModel {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            token_embedding_table: TokenEmbeddingTable::new(vocab_size, vocab_size),
        }
    }

    // pub fn forward(&self, idx: usize, target: usize) -> f64 {
    //     // idx and targets are both (B, T) tensors of integers
    //     let logits = self.token_embedding_table.get(idx); //(B, T, C)
    // }
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
