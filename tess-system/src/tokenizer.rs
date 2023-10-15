// This

use std::collections::HashMap;

use rayon::prelude::{ParallelIterator};

#[derive(Clone, Debug)]
pub struct Token(u8);

// impl from Vec<Token> to Vec<u8>
impl From<Token> for u8 {
    fn from(token: Token) -> Self {
        token.0
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// u8 has a max value of 255
// our alphabet is 65 characters
pub struct Tokenizer {
    pub mapping: HashMap<String, Token>,
}

pub trait encode {
    fn encode(&self, input: String) -> Result<Vec<Token>, String>;
}
pub trait decode {
    fn decode(&self, input: Vec<Token>) -> Result<String, String>;
}

impl Tokenizer {
    pub fn new(vocab: String) -> Self {
        // create a hashmap from the vocab
        let mut mapping = HashMap::new();
        for (i, c) in vocab.chars().enumerate() {
            mapping.insert(c.to_string(), Token(i as u8));
        }

        Self { mapping }
    }
}

impl encode for Tokenizer {
    fn encode(&self, input: String) -> Result<Vec<Token>, String> {
        let mut output = Vec::new();
        for c in input.chars() {
            match self.mapping.get(&c.to_string()) {
                Some(token) => output.push(token.clone()),
                None => return Err(format!("Error encoding character '{}'", c)),
            }
        }
        Ok(output)
    }
}
impl decode for Tokenizer {
    fn decode(&self, input: Vec<Token>) -> Result<String, String> {
        let mut output = String::new();
        for t in input {
            match self.mapping.iter().find(|(_, v)| v.0 == t.0) {
                Some((k, _)) => output.push_str(k),
                None => return Err(format!("Error decoding token '{:?}'", t)),
            }
        }
        Ok(output)
    }
}

// lets write some tests for the tokenizer
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let vocab = String::from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ");
        let tokenizer = Tokenizer::new(vocab);

        let input = String::from("Hello World!");
        let encoded = tokenizer.encode(input.clone()).unwrap();
        let decoded = tokenizer.decode(encoded).unwrap();

        assert_eq!(input, decoded);
    }
}
