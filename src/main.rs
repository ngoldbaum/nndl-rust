use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct MnistData {
    images: Vec<Vec<f32>>,
    classification: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dbg!(load_data(Path::new("test_data.json.gz"))?);
    Ok(())
}

fn load_data(path: &Path) -> Result<MnistData, std::io::Error> {
    let f = File::open(path)?;
    let mut gz = GzDecoder::new(f);
    let mut contents = String::new();
    gz.read_to_string(&mut contents)?;
    let data: MnistData = serde_json::from_str(&contents)?;
    Ok(data)
}
