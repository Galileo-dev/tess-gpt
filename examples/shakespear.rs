use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use rayon::prelude::*;
use tess_math::tensor::Tensor;
use tess_system::tokenizer::{encode, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // start by loading the shakespear text
    let filename = r"dataset\shakespear\input.txt";
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // print the length of the text
    println!(
        "Length of characters: {}",
        reader.get_ref().metadata()?.len()
    );

    // get a vec of character.
    // treat new lines as \n delimiters
    let chars: Vec<char> = reader
        .lines()
        .flat_map(|l| {
            l.unwrap()
                .chars()
                .chain(std::iter::once('\n'))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut chars_unique = chars.clone();
    chars_unique.par_sort();
    chars_unique.dedup();

    println!("number of unique characters: {}", chars_unique.len());

    // vocab is a string of all the unique characters
    let vocab = chars_unique.iter().collect::<String>();

    // create a tokenizer
    let tokenizer = Tokenizer::new(vocab);

    // print the mapping the tokenizer uses
    println!("tokenizer mapping:");
    let tokenizer_chars = tokenizer.mapping.keys().collect::<Vec<_>>();

    let tokenizer_tokens = tokenizer.mapping.values().collect::<Vec<_>>();

    print!("tokenizer chars: ");
    for c in tokenizer_chars {
        if c == "\n" {
            print!("\\n ");
        } else {
            print!("{} ", c);
        }
    }
    println!();
    print!("tokenizer tokens: ");
    for t in tokenizer_tokens {
        print!("{} ", t);
    }
    println!();
    println!();
    // use chars to create a string and encode it
    let input = chars.iter().collect::<String>();
    let encoded = tokenizer.encode(input.clone())?;

    // print the first 100 characters
    println!("Text: ");
    for c in &chars[0..100] {
        print!("{c}");
    }
    println!();

    // print the first 100 tokens
    print!("Tokens: ");
    for t in &encoded[0..100] {
        print!("{t} ");
    }
    println!();

    // convert from a vec of tokens to a vec of u8
    let encoded: Vec<u8> = encoded.into_iter().map(|t| t.into()).collect();

    // create a tensor from the encoded tokens
    let shape = vec![encoded.clone().len()];
    let data = Tensor::new(encoded, shape);

    // get the first 100 tokens from the tensor
    // slice of usize from 0 to 100
    // let data = data.get(slice);
    println!("index 1 is {}", data.get(&[1]));
    println!("the first 100:  {:?}", data.get_range(&[0..100]));

    // train test split of 90% train and 10% test
    let n = (data.len() as f32 * 0.9) as usize; // first 90% of the data
    let train = data.get_range(&[..n]);
    let test = data.get_range(&[n..]);
    println!("train len: {}", train.len());
    println!("test len: {}", test.len());

    Ok(())
}
