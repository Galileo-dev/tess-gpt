use rand::Rng;
use tess_system::tokenizer::{decode, encode, Token, Tokenizer};
fn main() {
    benchmark_tokenizer();
}

fn benchmark_tokenizer() {
    let vocab: String;
    let vocab_size: usize;
    let tokenizer: Tokenizer;
    let input: String;
    let encoded: Vec<Token>;
    let decoded: String;
    let start: std::time::Instant;
    let end: std::time::Instant;

    println!("-------------------------");
    println!("Benchmarking tokenizer...");
    println!("-------------------------");

    println!("01 - 1000 iter encode/decode of 11 chars");
    tokenizer_iter_char(1000, 11, None);
    println!("-------------------------");
    println!("02 - 1,000,000 iter encode/decode of 10,000 chars");
    tokenizer_iter_char(1_000_000, 10000, None);
}

fn tokenizer_iter_char(iterations: usize, input_size: usize, vocab: Option<String>) {
    // if no vocab is provided, use the default
    let vocab = vocab.map_or_else(|| String::from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "), |v| v);
    let vocab_size = vocab.len();

    let start = std::time::Instant::now();
    let tokenizer = Tokenizer::new(vocab);
    let end = std::time::Instant::now();

    println!("tokenizer created in {}ms...", (end - start).as_millis());

    // input is a random string of the specified size
    let input = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(input_size)
        .map(char::from)
        .collect::<String>();

    println!("input: {input}");
    print!("output: ");
    let encoded = tokenizer.encode(input.clone());
    for t in &encoded {
        print!("{t} ");
    }
    println!();

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let encoded = tokenizer.encode(input.clone());
        let decoded = tokenizer.decode(encoded);
    }
    let end = std::time::Instant::now();
    println!("Time taken: {}ms", (end - start).as_millis());
}
