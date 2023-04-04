use tess_system::tokenizer::{decode, encode, Tokenizer};
fn main() -> Result<(), String> {
    let vocab = String::from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ");
    let vocab_size = vocab.len();
    let tokenizer = Tokenizer::new(vocab);

    let input = String::from("Hello World!");
    let encoded = tokenizer.encode(input.clone())?;
    // print the encoded tokens
    print!("Tokens: ");
    for t in &encoded {
        print!("{t} ");
    }
    println!();
    let decoded = tokenizer.decode(encoded)?;
    println!("String: {decoded}");
    assert_eq!(input, decoded);

    Ok(())
}
