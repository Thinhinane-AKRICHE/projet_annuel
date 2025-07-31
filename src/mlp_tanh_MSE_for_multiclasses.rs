use rand::Rng;
use crate::loss::mse;
use std::collections::HashSet;

pub struct MLP {
    pub weights_hidden: Vec<Vec<f64>>,
    pub bias_hidden: Vec<f64>,
    pub weights_output: Vec<f64>,
    pub bias_output: f64,
    pub learning_rate: f64,
    pub epochs: usize,
    pub is_regression: bool,
    pub use_activation: bool,
}


//// Partie MLP pour la classification binaire et regresssion//////
impl MLP {
    pub fn new(n_inputs: usize, n_hidden: usize, learning_rate: f64, epochs: usize, is_regression: bool, use_activation: bool) -> Self {
        let mut rng = rand::thread_rng();

        let weights_hidden = (0..n_hidden)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let bias_hidden = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let weights_output = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = rng.gen_range(-1.0..1.0);

        Self {
            weights_hidden,
            bias_hidden,
            weights_output,
            bias_output,
            learning_rate,
            epochs,
            is_regression,
            use_activation,
        }
    }

    fn activate(&self, x: f64) -> f64 {
        if self.use_activation {
            x.tanh()
        } else {
            x
        }
    }

    fn activate_derivative(&self, x: f64) -> f64 {
        if self.use_activation {
            1.0 - x.tanh().powi(2)
        } else {
            1.0
        }
    }

    pub fn fit(&mut self, X: &Vec<Vec<f64>>, y: &Vec<f64>) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                let hidden_input: Vec<f64> = self.weights_hidden.iter()
                    .zip(self.bias_hidden.iter())
                    .map(|(w, &b)| w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                    .collect();

                let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

                let output_input = hidden_output.iter().zip(self.weights_output.iter()).map(|(h, w)| h * w).sum::<f64>() + self.bias_output;
                let output = if self.is_regression {
                    output_input
                } else {
                    self.activate(output_input)
                };

                let error = output - yi;
                let delta_output = if self.is_regression {
                    error
                } else {
                    error * self.activate_derivative(output_input)
                };

                for i in 0..self.weights_output.len() {
                    self.weights_output[i] -= self.learning_rate * delta_output * hidden_output[i];
                }
                self.bias_output -= self.learning_rate * delta_output;

                for j in 0..self.weights_hidden.len() {
                    let delta_hidden = delta_output * self.weights_output[j] * self.activate_derivative(hidden_input[j]);
                    for k in 0..self.weights_hidden[j].len() {
                        self.weights_hidden[j][k] -= self.learning_rate * delta_hidden * xi[k];
                    }
                    self.bias_hidden[j] -= self.learning_rate * delta_hidden;
                }
            }
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        let hidden_input: Vec<f64> = self.weights_hidden.iter()
            .zip(self.bias_hidden.iter())
            .map(|(w, &b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.activate(h)).collect();

        let output_input = hidden_output.iter()
            .zip(self.weights_output.iter())
            .map(|(h, w)| h * w)
            .sum::<f64>() + self.bias_output;

        if self.is_regression {
            output_input
        } else {
            self.activate(output_input)
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}
fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

fn mse_loss(output: &[f64], target: &[f64]) -> f64 {
    output.iter()
        .zip(target.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f64>() / output.len() as f64
}

fn one_hot_tanh(target: usize, n_classes: usize) -> Vec<f64> {
    (0..n_classes)
        .map(|i| if i == target { 1.0 } else { -1.0 })
        .collect()
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
}

//// Partie MLP pour la classification multiclasses//////


pub struct MLPDeepClassifier {
    pub batch_size: usize,
    pub weights_hidden: Vec<Vec<Vec<f64>>>,
    pub bias_hidden: Vec<Vec<f64>>,
    pub weights_output: Vec<Vec<f64>>,
    pub bias_output: Vec<f64>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
    pub activation: Activation,
    pub lambda: f64,
    // === Suivi loss + accuracy ===
    pub train_losses: Vec<f64>,
    pub test_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>,
    pub test_accuracies: Vec<f64>,
}

impl MLPDeepClassifier {
    pub fn new(
        batch_size: usize,
        n_inputs: usize,
        hidden_layers: Vec<usize>,
        n_classes: usize,
        learning_rate: f64,
        epochs: usize,
        activation: Activation,
        lambda: f64
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights_hidden = vec![];
        let mut bias_hidden = vec![];
        let mut prev_size = n_inputs;

        for &layer_size in &hidden_layers {
            weights_hidden.push(
                (0..layer_size)
                    .map(|_| (0..prev_size).map(|_| rng.gen_range(-0.05..0.05)).collect())
                    .collect()
            );
            bias_hidden.push((0..layer_size).map(|_| rng.gen_range(-1.0..1.0)).collect());
            prev_size = layer_size;
        }

        let weights_output = (0..n_classes)
            .map(|_| (0..prev_size).map(|_| rng.gen_range(-0.05..0.05)).collect())
            .collect();
        let bias_output = (0..n_classes).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            batch_size,
            weights_hidden,
            bias_hidden,
            weights_output,
            bias_output,
            learning_rate,
            epochs,
            n_classes,
            activation,
            lambda,
            train_losses: Vec::with_capacity(epochs),
            test_losses: Vec::with_capacity(epochs),
            train_accuracies: Vec::with_capacity(epochs),
            test_accuracies: Vec::with_capacity(epochs),
        }
    }



    pub fn fit(
    &mut self,
    X: &[Vec<f64>],
    y: &[usize],
    X_test: Option<&[Vec<f64>]>,
    y_test: Option<&[usize]>,
) {
    let n_samples = X.len();

    for epoch_index in 0..self.epochs {
        for (x_batch, y_batch) in X.chunks(self.batch_size).zip(y.chunks(self.batch_size)) {
            let batch_len = x_batch.len();

            // Initialisation des gradients
            let mut grad_weights_output = vec![vec![0.0; self.weights_output[0].len()]; self.n_classes];
            let mut grad_bias_output = vec![0.0; self.n_classes];
            let mut grad_weights_hidden = self.weights_hidden
                .iter()
                .map(|layer| vec![vec![0.0; layer[0].len()]; layer.len()])
                .collect::<Vec<_>>();
            let mut grad_bias_hidden = self.bias_hidden
                .iter()
                .map(|layer| vec![0.0; layer.len()])
                .collect::<Vec<_>>();

            for (xi, &yi) in x_batch.iter().zip(y_batch.iter()) {
                // === Forward pass ===
                let mut hidden_inputs = vec![];
                let mut hidden_outputs = vec![];
                let mut input = xi.clone();

                for (weights, biases) in self.weights_hidden.iter().zip(self.bias_hidden.iter()) {
                    let z: Vec<f64> = weights.iter()
                        .zip(biases.iter())
                        .map(|(w, &b)| w.iter().zip(&input).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                        .collect();
                    let h: Vec<f64> = z.iter().map(|&x| match self.activation {
                        Activation::ReLU => relu(x),
                        Activation::Tanh => tanh(x),
                    }).collect();
                    hidden_inputs.push(z);
                    hidden_outputs.push(h.clone());
                    input = h;
                }

                // === Sortie ===
                let logits: Vec<f64> = self.weights_output.iter()
                    .zip(&self.bias_output)
                    .map(|(w, &b)| w.iter().zip(&input).map(|(wi, hi)| wi * hi).sum::<f64>() + b)
                    .collect();

                let output: Vec<f64> = logits.iter().map(|&z| tanh(z)).collect();
                let target = one_hot_tanh(yi, self.n_classes);

                // === Erreur de sortie ===
                let mut delta_output = vec![0.0; self.n_classes];
                for j in 0..self.n_classes {
                    let grad = 1.0 - output[j].powi(2); // dérivée tanh
                    delta_output[j] = 2.0 * (output[j] - target[j]) * grad;
                }

                // === Gradients couche de sortie ===
                for j in 0..self.n_classes {
                    for k in 0..self.weights_output[j].len() {
                        grad_weights_output[j][k] += delta_output[j] * input[k];
                    }
                    grad_bias_output[j] += delta_output[j];
                }

                // === Backpropagation dans les couches cachées ===
                let mut delta = delta_output.clone();
                for l in (0..self.weights_hidden.len()).rev() {
                    let mut new_delta = vec![0.0; self.weights_hidden[l].len()];

                    for j in 0..self.weights_hidden[l].len() {
                        let mut error = 0.0;
                        for k in 0..delta.len() {
                            let w = if l == self.weights_hidden.len() - 1 {
                                self.weights_output[k][j]
                            } else {
                                self.weights_hidden[l + 1][k][j]
                            };
                            error += delta[k] * w;
                        }

                        let grad = match self.activation {
                            Activation::ReLU => relu_derivative(hidden_inputs[l][j]),
                            Activation::Tanh => tanh_derivative(hidden_inputs[l][j]),
                        };
                        new_delta[j] = error * grad;

                        for i in 0..(if l == 0 { xi.len() } else { hidden_outputs[l - 1].len() }) {
                            let input_val = if l == 0 { xi[i] } else { hidden_outputs[l - 1][i] };
                            grad_weights_hidden[l][j][i] += new_delta[j] * input_val;
                        }
                        grad_bias_hidden[l][j] += new_delta[j];
                    }

                    delta = new_delta;
                }
            }

            // === Mise à jour des poids ===
            let scale = self.learning_rate / batch_len as f64;

            for j in 0..self.n_classes {
                for k in 0..self.weights_output[j].len() {
                    self.weights_output[j][k] -= scale * (grad_weights_output[j][k] + self.lambda * self.weights_output[j][k]);
                }
                self.bias_output[j] -= scale * grad_bias_output[j];
            }

            for l in 0..self.weights_hidden.len() {
                for j in 0..self.weights_hidden[l].len() {
                    for i in 0..self.weights_hidden[l][j].len() {
                        self.weights_hidden[l][j][i] -= scale * (grad_weights_hidden[l][j][i] + self.lambda * self.weights_hidden[l][j][i]);
                    }
                    self.bias_hidden[l][j] -= scale * grad_bias_hidden[l][j];
                }
            }
        }

        // === Fin de l'époque : évaluation
        let train_loss = self.compute_loss(X, y);
        let train_acc = self.accuracy(X, y);

        self.train_losses.push(train_loss);
        self.train_accuracies.push(train_acc);

        if let (Some(Xt), Some(yt)) = (X_test, y_test) {
            self.test_losses.push(self.compute_loss(Xt, yt));
            self.test_accuracies.push(self.accuracy(Xt, yt));
        }

        if epoch_index % 10 == 0 || epoch_index == self.epochs - 1 {
            println!(
                "Epoch {}/{} — Loss: {:.4} — Train Acc: {:.2}% — Test Acc: {:.2}%",
                epoch_index + 1,
                self.epochs,
                train_loss,
                train_acc * 100.0,
                self.test_accuracies.last().unwrap_or(&0.0) * 100.0
            );
        }
    }

    println!(" [MLP] Entraînement terminé !");
}


    pub fn compute_loss(&self, X: &[Vec<f64>], y: &[usize]) -> f64 {
        let mut loss = 0.0;
        for (xi, &yi) in X.iter().zip(y.iter()) {
            let output = self.predict_output(xi);
            let target = one_hot_tanh(yi, self.n_classes);
            for j in 0..self.n_classes {
                let err = output[j] - target[j];
                loss += err * err;
            }
        }
        loss / X.len() as f64
    }

    pub fn accuracy(&self, X: &[Vec<f64>], y: &[usize]) -> f64 {
        let mut correct = 0;
        for (xi, &yi) in X.iter().zip(y.iter()) {
            if self.predict(xi) == yi {
                correct += 1;
            }
        }
        correct as f64 / X.len() as f64
    }

    pub fn predict_output(&self, x: &[f64]) -> Vec<f64> {
        let mut input = x.to_vec();
        for (weights, biases) in self.weights_hidden.iter().zip(self.bias_hidden.iter()) {
            let z: Vec<f64> = weights.iter()
                .zip(biases.iter())
                .map(|(w, &b)| w.iter().zip(&input).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                .collect();
            input = z.iter().map(|&x| match self.activation {
                Activation::ReLU => relu(x),
                Activation::Tanh => tanh(x),
            }).collect();
        }

        let logits: Vec<f64> = self.weights_output.iter()
            .zip(&self.bias_output)
            .map(|(w, &b)| w.iter().zip(&input).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        logits.iter().map(|&z| tanh(z)).collect()
    }

    pub fn predict(&self, x: &[f64]) -> usize {
        let mut input = x.to_vec();

        for (weights, biases) in self.weights_hidden.iter().zip(self.bias_hidden.iter()) {
            let z: Vec<f64> = weights.iter()
                .zip(biases.iter())
                .map(|(w, &b)| w.iter().zip(&input).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                .collect();
            input = z.iter().map(|&x| match self.activation {
                Activation::ReLU => relu(x),
                Activation::Tanh => tanh(x),
            }).collect();
        }

        let logits: Vec<f64> = self.weights_output.iter()
            .zip(&self.bias_output)
            .map(|(w, &b)| w.iter().zip(&input).map(|(wi, hi)| wi * hi).sum::<f64>() + b)
            .collect();

        let output: Vec<f64> = logits.iter().map(|&z| tanh(z)).collect();

        output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

}
