// ===========================================================
// This example is based on "Let's build GPT: from scratch, in
// code, spelled out" by Andrej Karpathy.
// For more information, see the video:
// https://www.youtube.com/watch?v=kCc8FmEb1nY
// ===========================================================

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use rayon::prelude::*;
use tess_math::math::Tensor;
use tess_system::tokenizer::{encode, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // start by loading the shakespear text
    let filename = r"./dataset/shakespear.txt";
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
            print!("{c} ");
        }
    }
    println!();
    print!("tokenizer tokens: ");
    for t in tokenizer_tokens {
        print!("{t} ");
    }
    println!();
    println!();
    // use chars to create a string and encode it
    let input = chars.iter().collect::<String>();
    let encoded = tokenizer.encode(input)?;

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
    let shape = vec![1, encoded.clone().len()];
    println!("shape: {:?}", shape);
    let data = Tensor::new(encoded, shape);

    // get the first 100 tokens from the tensor
    // slice of usize from 0 to 100
    // let data = data.get(slice);
    // println!("data: {:?}", data);
    println!("index 1 is {:?}", data.get(.., 1));
    println!("the first 100:  {:?}", data.get(.., 0..100));

    // train test split of 90% train and 10% test
    let n = (data.len() as f32 * 0.9) as usize; // first 90% of the data
    let train_data = data.get(.., ..n);
    let test_data = data.get(.., n..);
    println!("train len: {}", train_data.len());
    println!("test len: {}", test_data.len());

    let block_size = 8; // 8 tokens per block
    let block = train_data.get(.., ..=block_size);
    println!("block: {block:?}");

    // show an example of how it learns
    let x = train_data.get(.., ..block_size);
    let y = train_data.get(.., 1..=block_size);
    for t in 0..block_size {
        let context = x.get(.., ..=t);
        let target = y.get(.., t);
        println!("when input is : {context:?},\n the target is : {target:?}");
    }

    // batch dimension
    let batch_size: usize = 4; // how many independent sequences we want to train on in sequence
    let block_size: usize = 8; // maximum context length for prediction

    // get a batch of data
    let (xb, yb) = get_batch("train", &test_data, &train_data, batch_size, block_size);

    println!("inputs: {xb:?}");
    println!("targets: {yb:?}");

    println!("-----");

    for b in 0..batch_size {
        println!("batch: {b}");
        for t in 0..block_size {
            let context = xb.get(b, 0..=t);
            let target = yb.get(b, t);
            println!("when input is : {context:?}, the target is : {target:?}");
        }
    }

    Ok(())
}

fn get_batch(
    split: &str,
    test_data: &Tensor<u8>,
    train_data: &Tensor<u8>,
    batch_size: usize,
    block_size: usize,
) -> (Tensor<u8>, Tensor<u8>) {
    let data = match split {
        "train" => train_data,
        "test" => test_data,
        _ => panic!("split must be 'train' or 'test'"),
    };

    let ix: Tensor<i32> = Tensor::<i32>::randn(
        0,
        (data.len() - block_size).try_into().unwrap(),
        vec![batch_size],
        1337,
    );

    println!("ix: {ix:?}");
    println!("data shape: {:?}", data.shape());
    println!("data: {:?}", data.get(.., 0..100));

    // [data[i:i+block_size] for i in ix]
    let x = ix
        .iter()
        .map(|i| {
            let adjusted_index = *i as usize;

            data.get(.., adjusted_index..adjusted_index + block_size)
        })
        .collect::<Tensor<u8>>();

    // [data[i+1:i+block_size+1] for i in ix]
    let y = ix
        .iter()
        .map(|i| {
            let adjusted_index = *i as usize;

            data.get(.., (adjusted_index + 1)..=(adjusted_index + block_size))
        })
        .collect::<Tensor<u8>>();

    (x, y)
}
