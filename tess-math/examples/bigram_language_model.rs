use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use tess_math::language_models::{BigramLanguageModel, LanguageModel, LanguageModelDataset};

fn main() {
    println!("=========================");
    println!("= Bigram Language Model =");
    println!("=========================");

    // start by loading the names dataset
    let filename = r"../dataset/names/input.txt";

    // read the file into a vector of strings
    let names = read_file(filename);

    // print the length of the text
    println!("Length of names: {}", names.len());

    // create a dataset from the names
    let dataset = LanguageModelDataset { data: names };

    // now create our bigram language model
    // let mut bigram_model = BigramLanguageModel::new();

    // train the model
    bigram_model.train(&dataset);
}

fn read_file(filename: &str) -> Vec<String> {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    // read the file line by line
    let mut names = Vec::new();
    for line in reader.lines() {
        let name = line.unwrap();
        names.push(name);
    }

    names
}
