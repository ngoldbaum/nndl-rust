#[macro_use(array)]
extern crate ndarray;

#[macro_use(lazy_static)]
extern crate lazy_static;

use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;
use rand::seq::SliceRandom;
use rand::thread_rng;

mod mnist_reader;

type MnistImage = mnist_reader::MnistImage;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

lazy_static! {
    static ref TRUE_ACTIVATIONS: [Array2<f64>; 10] = [
        array![[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]
            .t()
            .to_owned(),
        array![[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
            .t()
            .to_owned(),
    ];
}

impl Network {
    fn new(sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();
        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), StandardNormal));
            weights.push(Array::random((sizes[i], sizes[i - 1]), StandardNormal));
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

    fn evaluate(&self, test_data: &[MnistImage]) -> usize {
        let test_results = test_data
            .iter()
            .map(|x| self.feedforward(&x.image))
            .map(|x| argmax(&x))
            .collect::<Vec<usize>>();
        test_results
            .iter()
            .zip(test_data.iter())
            .map(|(x, y)| (*x == y.classification as usize) as usize)
            .sum()
    }

    fn sgd(
        &mut self,
        training_data: &[MnistImage],
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: &[MnistImage],
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
        training_data: &[MnistImage],
        mini_batch_indices: &[usize],
        eta: f64,
    ) {
        let mut nabla_b: Vec<Array2<f64>> = zero_vec_like(&self.biases);
        let mut nabla_w: Vec<Array2<f64>> = zero_vec_like(&self.weights);
        for i in mini_batch_indices {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(&training_data[*i]);
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
        }
        let nbatch = mini_batch_indices.len() as f64;
        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            *w -= &(nw.mapv(|x| x * eta / nbatch));
        }
        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            *b -= &(nb.mapv(|x| x * eta / nbatch));
        }
    }

    fn backprop(&self, data: &MnistImage) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::zeros(to_tuple(b.shape())))
            .collect();
        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::zeros(to_tuple(w.shape())))
            .collect();
        let mut activation = data.image.clone();
        let mut activations = vec![activation.clone()];
        let mut zs: Vec<Array2<f64>> = Vec::new();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(&activation) + b;
            activation = sigmoid(&z);
            zs.push(z);
            activations.push(activation.clone());
        }
        let delta = self.cost_derivative(activations.last().unwrap(), data.classification as usize)
            * sigmoid_prime(zs.last().unwrap());
        let nbiases = self.biases.len();
        let nweights = self.weights.len();
        nabla_b[nbiases - 1] = delta.clone();
        nabla_w[nbiases - 1] = delta.dot(&activations[activations.len() - 2].t());
        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z);
            let delta = self.weights[nweights - l + 1].t().dot(&delta) * sp;
            nabla_b[nbiases - l] = delta.clone();
            nabla_w[nweights - l] = delta.dot(&activations[activations.len() - l - 1].t());
        }
        (nabla_b, nabla_w)
    }

    fn cost_derivative(&self, output_activations: &Array2<f64>, y: usize) -> Array2<f64> {
        output_activations - &TRUE_ACTIVATIONS[y]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = mnist_reader::load_data("t10k")?;
    let training_data = mnist_reader::load_data("train")?;
    let mut net = Network::new(&[784, 30, 10]);
    net.sgd(&training_data, 30, 20, 3.0, &test_data);
    Ok(())
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

fn zero_vec_like(inp: &[Array2<f64>]) -> Vec<Array2<f64>> {
    inp.iter()
        .map(|x| Array2::zeros(to_tuple(x.shape())))
        .collect()
}
