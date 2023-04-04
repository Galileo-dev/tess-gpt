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

fn tokenizer_iter_char(
    iterations: usize,
    input_size: usize,
    vocab: Option<String>,
) -> Result<(), String> {
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

    let output = tokenizer.encode(input.clone())?;
    // cut off the end of the string if it is too long and add a number to the end to indicate how many chars were cut off
    let input = if input.len() > 100 {
        let mut input = input.chars().take(100).collect::<String>();
        input.push_str(&format!("...({} chars cut off)", input_size - 100));
        input
    } else {
        input
    };

    // cut off the end of the string if it is too long and add a number to the end to indicate how many chars were cut off
    let output_info: String = if output.len() > 100 {
        let mut output = output
            .iter()
            .take(100)
            .map(|t| t.to_string())
            .collect::<String>();
        output.push_str(&format!("...({} tokens cut off)", output.len() - 100));
        output
    } else {
        output.iter().map(|t| t.to_string()).collect::<String>()
    };

    println!("input: {input}");
    println!("output: {output_info}");

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let encoded = tokenizer.encode(input.clone())?;
        let decoded = tokenizer.decode(encoded);
    }
    let end = std::time::Instant::now();
    println!("Time taken: {}ms", (end - start).as_millis());

    Ok(())
}
