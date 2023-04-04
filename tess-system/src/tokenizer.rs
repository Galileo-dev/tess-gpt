// This

use std::collections::HashMap;

#[derive(Clone)]
pub struct Token(u8);

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// u8 has a max value of 255
// our alphabet is 65 characters

pub struct Tokenizer {
    mapping: HashMap<String, Token>,
}

pub trait encode {
    fn encode(&self, input: String) -> Vec<Token>;
}
pub trait decode {
    fn decode(&self, input: Vec<Token>) -> String;
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
    fn encode(&self, input: String) -> Vec<Token> {
        let mut output = Vec::new();
        for c in input.chars() {
            output.push(self.mapping.get(&c.to_string()).unwrap().clone());
        }
        output
    }
}

impl decode for Tokenizer {
    fn decode(&self, input: Vec<Token>) -> String {
        let mut output = String::new();
        for t in input {
            for (k, v) in &self.mapping {
                if v.0 == t.0 {
                    output.push_str(k);
                }
            }
        }
        output
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
        let encoded = tokenizer.encode(input.clone());
        let decoded = tokenizer.decode(encoded);

        assert_eq!(input, decoded);
    }
}
