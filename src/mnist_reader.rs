use flate2::read::GzDecoder;
use ndarray::{Array, Array2};
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize, Debug)]
struct MnistDataRead {
    images: Vec<Vec<f64>>,
    classification: Vec<usize>,
}

#[derive(Debug)]
pub struct MnistData {
    pub image: Array2<f64>,
    pub classification: usize,
}

pub fn load_data(path: &Path) -> Result<Vec<MnistData>, std::io::Error> {
    println!("Loading {:?}", &path);
    let f = File::open(path)?;
    let mut gz = GzDecoder::new(f);
    let mut contents = String::new();
    gz.read_to_string(&mut contents)?;
    let data: MnistDataRead = serde_json::from_str(&contents)?;
    let data: Vec<MnistData> = data
        .images
        .into_iter()
        .zip(data.classification.iter())
        .map(|(x, y)| MnistData {
            image: Array::from_shape_vec((x.len(), 1), x).unwrap(),
            classification: *y,
        })
        .collect();
    Ok(data)
}
