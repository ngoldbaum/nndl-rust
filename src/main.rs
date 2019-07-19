use flate2::read::GzDecoder;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::thread_rng;
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
    image: Array2<f64>,
    classification: usize,
}

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    pub biases: Vec<Array2<f64>>,
    pub weights: Vec<Array2<f64>>,
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

    fn evaluate(&self, test_data: &[MnistData]) -> usize {
        let test_results = test_data
            .iter()
            .map(|x| self.feedforward(&x.image))
            .map(|x| argmax(&x))
            .collect::<Vec<usize>>();
        test_results
            .iter()
            .zip(test_data.iter())
            .map(|(x, y)| (*x == y.classification) as usize)
            .sum()
    }

    fn SGD(
        &mut self,
        training_data: &[MnistData],
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: &[MnistData],
    ) {
        let n_test = test_data.len();
        let n = training_data.len();
        let mut indices = (0..n).collect::<Vec<usize>>();
        for j in 0..epochs {
            indices.shuffle(&mut thread_rng());
            for sl in (0..n)
                .step_by(mini_batch_size)
                .collect::<Vec<usize>>()
                .windows(2)
            {
                self.update_mini_batch(training_data, &indices[sl[0]..sl[1]], eta);
            }
            println!("Epoch {}: {} / {}", j, self.evaluate(test_data), n_test);
        }
    }

    fn update_mini_batch(
        &mut self,
        training_data: &[MnistData],
        mini_batch_indices: &[usize],
        eta: f64,
    ) {
        let nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::zeros(to_tuple(b.shape())))
            .collect();
        let nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::zeros(to_tuple(w.shape())))
            .collect();
        for i in mini_batch_indices {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(&training_data[*i]);
            let nabla_b: Vec<Array2<f64>> = nabla_b
                .iter()
                .zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            let nabla_w: Vec<Array2<f64>> = nabla_w
                .iter()
                .zip(delta_nabla_w.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }
        let nbatch = mini_batch_indices.len() as f64;
        self.weights = self
            .weights
            .iter()
            .cloned()
            .zip(nabla_w.iter().map(|nw| ((eta / nbatch) * nw)))
            .map(|(w, f)| w - f)
            .collect();
        self.biases = self
            .biases
            .iter()
            .cloned()
            .zip(nabla_b.iter().map(|nb| ((eta / nbatch) * nb)))
            .map(|(b, f)| b - f)
            .collect()
    }

    fn backprop(&self, data: &MnistData) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::zeros(to_tuple(b.shape())))
            .collect();
        let nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::zeros(to_tuple(w.shape())))
            .collect();
        (nabla_b, nabla_w)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_data(Path::new("test_data.json.gz"))?;
    let net = Network::new(&[784, 30, 10]);
    dbg!(net.evaluate(&data));
    Ok(())
}

fn load_data(path: &Path) -> Result<Vec<MnistData>, std::io::Error> {
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

fn to_tuple(inp: &[usize]) -> (usize, usize) {
    match inp {
        [a, b] => (*a, *b),
        _ => panic!(),
    }
}
