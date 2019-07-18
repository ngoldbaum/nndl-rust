use flate2::read::GzDecoder;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
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

struct MnistData {
    images: Vec<Array2<f64>>,
    classification: Vec<usize>,
}

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();
        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), Uniform::new(0., 1.)));
            weights.push(Array::random(
                (sizes[i], sizes[i - 1]),
                Uniform::new(0., 1.),
            ));
        }
        Network {
            num_layers: num_layers,
            sizes: sizes.to_owned(),
            biases: biases,
            weights: weights,
        }
    }

    fn feedforward(&self, a: &Array2<f64>) -> Array2<f64> {
        let mut ret = a.clone();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            ret = sigmoid(&(w.dot(&ret) + b));
        }
        ret
    }

    fn evaluate(&self, test_data: &MnistData) -> usize {
        let test_results = test_data
            .images
            .iter()
            .map(|x| self.feedforward(x))
            .map(|x| argmax(&x))
            .collect::<Vec<usize>>();
        test_results
            .iter()
            .zip(test_data.classification.iter())
            .map(|(x, y)| (x == y) as usize)
            .sum()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_data(Path::new("test_data.json.gz"))?;
    let net = Network::new(&[784, 30, 10]);
    dbg!(net.evaluate(&data));
    Ok(())
}

fn load_data(path: &Path) -> Result<MnistData, std::io::Error> {
    let f = File::open(path)?;
    let mut gz = GzDecoder::new(f);
    let mut contents = String::new();
    gz.read_to_string(&mut contents)?;
    let data: MnistDataRead = serde_json::from_str(&contents)?;
    let data: MnistData = MnistData {
        images: data
            .images
            .into_iter()
            .map(|x| Array::from_shape_vec((x.len(), 1), x).unwrap())
            .collect::<Vec<Array2<f64>>>(),
        classification: data.classification,
    };
    Ok(data)
}

fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-z).mapv(f64::exp))
}

fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn argmax(a: &Array2<f64>) -> usize {
    let mut ret = 0;
    for (i, el) in a.iter().enumerate() {
        if *el > a[[ret, 0]] {
            ret = i;
        }
    }
    ret
}
