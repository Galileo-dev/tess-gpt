mod bigram_model;
pub use bigram_model::{BigramLanguageModel, LanguageModelDataset};

mod token_embedding_table;
pub use token_embedding_table::TokenEmbeddingTable;

use std::fmt::{Debug, Display};

pub trait LanguageModel<T>
where
    T: Debug + Display,
{
    fn train(&mut self, dataset: &LanguageModelDataset<T>);
    fn score(&self, dataset: &LanguageModelDataset<T>) -> f64;
}
