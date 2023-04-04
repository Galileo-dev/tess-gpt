use tess_system::tokenizer::{decode, encode, Tokenizer};
fn main() {
    let vocab = String::from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ");
    let vocab_size = vocab.len();
    let tokenizer = Tokenizer::new(vocab);

    let input = String::from("Hello World!");
    let encoded = tokenizer.encode(input.clone());
    // print the encoded tokens
    print!("Tokens: ");
    for t in &encoded {
        print!("{t} ");
    }
    println!();
    let decoded = tokenizer.decode(encoded);
    println!("String: {decoded}");
    assert_eq!(input, decoded);

    println!("-------------------------");
    println!("Benchmarking tokenizer...");
    println!("-------------------------");
    println!("01 - 1000 encode/decode");
    println!("Vocab size: {vocab_size}");
    println!("Input size: {}", input.len());

    // benchmark the tokenizer
    let input = String::from("Hello World!");
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let encoded = tokenizer.encode(input.clone());
        let decoded = tokenizer.decode(encoded);
    }
    let end = std::time::Instant::now();
    println!("Time taken: {}ms", (end - start).as_millis());
}
