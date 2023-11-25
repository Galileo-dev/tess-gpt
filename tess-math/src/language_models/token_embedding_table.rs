use crate::math::Tensor;

pub struct TokenEmbeddingTable {
    table: Tensor<u8>,
}

impl TokenEmbeddingTable {
    pub fn new(width: u32, height: u32) -> Self {
        // create a vec of zeros of size width * height
        let data: Tensor<u8> = Tensor::randn(0, 100, vec![3, 2], 1337);

        Self { table: data }
    }

    // pub fn forward(&self, idx: Tensor<u8>) -> Tensor<u8> {
    //     let batch_size = idx.shape()[0];
    //     let embedding_dim = self.table.shape()[1];
    //     let seq_len = idx.shape()[1];

    //     let mut output_tensor = Tensor::zeros(vec![batch_size, embedding_dim]);

    //     for i in 0..batch_size {
    //         for j in 0..seq_len {
    //             let token = idx.get(i, j);

    //             let embedding = self.table.get(token, ..);

    //             output_tensor.set(i, j, embedding);
    //         }
    //     }

    //     self.table.get(idx)
    // }
}
