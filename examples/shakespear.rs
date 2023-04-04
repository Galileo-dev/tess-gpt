fn main() {
    // start by loading the shakespear text
    let shakespear = std::fs::read_to_string(r"dataset\shakespear\input.txt").unwrap();

    // print the length of the text
    println!("Length of characters: {}", shakespear.len());

    // get a vec of unique characters
    let mut chars: Vec<char> = shakespear.chars().collect();
    chars.sort();
    chars.dedup();
    println!("Unique characters: {}", chars.len());
    // print the unique characters
    for c in &chars {
        print!("{} ", c);
    }

    let vocab_size = chars.len();

    // get a line by line iterator
    let lines = shakespear.lines();
    for line in lines {
        // print the line
    }
}
