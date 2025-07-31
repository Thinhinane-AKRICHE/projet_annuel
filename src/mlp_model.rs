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

//regression + classeification binaire 
impl MLP {
    pub fn new(n_inputs: usize, n_hidden: usize, learning_rate: f64, epochs: usize,is_regression: bool,use_activation: bool) -> Self {
        let mut rng = rand::thread_rng();

        let weights_hidden = (0..n_hidden)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        println!("Poids initiaux cachés : {:?}", weights_hidden);

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
                // Forward pass
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

                // Backward pass
                let error = output - yi;
                let delta_output = if self.is_regression {
                    error // dérivée de l'identité = 1
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
            self.activate(output_input) // tanh sinon
        }
    }

}

//partie classification multiclasses
// === Fichier : mlp_classifier.rs ===

// === Fonctions d’activation ===

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

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // stabilisation
    let exps: Vec<f64> = logits.iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum: f64 = exps.iter().sum();

    if sum == 0.0 || !sum.is_finite() {
        eprintln!("[Softmax] Somme invalide : {sum}");
        return vec![1.0 / logits.len() as f64; logits.len()];
    }

    exps.iter().map(|&x| x / sum).collect()
}



pub fn cross_entropy_loss(probs: &[f64], target: usize) -> f64 {
    let epsilon = 1e-15;
    let clipped = probs[target].max(epsilon);
    -clipped.ln()
}



// === Activation disponible ===
#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
}


// === MLP Classifier ===

pub struct MLPClassifier {
    pub batch_size: usize,
    pub weights_hidden: Vec<Vec<f64>>,
    pub bias_hidden: Vec<f64>,
    pub weights_output: Vec<Vec<f64>>, // [n_classes][n_hidden]
    pub bias_output: Vec<f64>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
    pub activation: Activation,
    pub lambda: f64, // Coefficient de régularisation L2
}

impl MLPClassifier {
    pub fn new(
        batch_size: usize,
        n_inputs: usize,
        n_hidden: usize,
        n_classes: usize,
        learning_rate: f64,
        epochs: usize,
        activation: Activation,
        lambda: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let weights_hidden = (0..n_hidden)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(-0.05..0.05)).collect())
            .collect();
        let bias_hidden = (0..n_hidden).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let weights_output = (0..n_classes)
            .map(|_| (0..n_hidden).map(|_| rng.gen_range(-0.05..0.05)).collect())
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
        }
    }

    fn apply_activation(&self, x: f64) -> f64 {
        match self.activation {
            Activation::ReLU => relu(x),
            Activation::Tanh => tanh(x),
        }
    }

    fn apply_activation_derivative(&self, x: f64) -> f64 {
        match self.activation {
            Activation::ReLU => relu_derivative(x),
            Activation::Tanh => tanh_derivative(x),
        }
    }

    pub fn fit(&mut self, X: &[Vec<f64>], y: &[usize]) {
        let n_samples = X.len();

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            for (x_batch, y_batch) in X.chunks(self.batch_size).zip(y.chunks(self.batch_size)) {
                let batch_len = x_batch.len();

                let mut grad_weights_output = vec![vec![0.0; self.weights_output[0].len()]; self.n_classes];
                let mut grad_bias_output = vec![0.0; self.n_classes];
                let mut grad_weights_hidden = vec![vec![0.0; self.weights_hidden[0].len()]; self.weights_hidden.len()];
                let mut grad_bias_hidden = vec![0.0; self.bias_hidden.len()];

                for (xi, &yi) in x_batch.iter().zip(y_batch.iter()) {
                    let hidden_input: Vec<f64> = self.weights_hidden.iter()
                        .zip(self.bias_hidden.iter())
                        .map(|(w, &b)| w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
                        .collect();

                    let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.apply_activation(h)).collect();

                    let logits: Vec<f64> = self.weights_output.iter()
                        .zip(self.bias_output.iter())
                        .map(|(w_out, &b)| hidden_output.iter().zip(w_out.iter()).map(|(h, w)| h * w).sum::<f64>() + b)
                        .collect();

                    let probs = softmax(&logits);
                    if probs.iter().any(|p| !p.is_finite()) {
                        eprintln!("[MLPClassifier] Probs contient NaN ou inf : {:?}", probs);
                        continue;
                    }

                    total_loss += cross_entropy_loss(&probs, yi);

                    let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(idx, _)| idx).unwrap_or(0);
                    if pred == yi {
                        correct += 1;
                    }

                    let mut delta_output = vec![0.0; self.n_classes];
                    for j in 0..self.n_classes {
                        delta_output[j] = probs[j] - if j == yi { 1.0 } else { 0.0 };
                    }

                    for j in 0..self.n_classes {
                        for k in 0..self.weights_output[j].len() {
                            grad_weights_output[j][k] += delta_output[j] * hidden_output[k];
                        }
                        grad_bias_output[j] += delta_output[j];
                    }

                    for h in 0..self.weights_hidden.len() {
                        let mut grad = 0.0;
                        for j in 0..self.n_classes {
                            grad += delta_output[j] * self.weights_output[j][h];
                        }
                        let delta_h = grad * self.apply_activation_derivative(hidden_input[h]);

                        for i in 0..self.weights_hidden[h].len() {
                            grad_weights_hidden[h][i] += delta_h * xi[i];
                        }
                        grad_bias_hidden[h] += delta_h;
                    }
                }

                let scale = self.learning_rate / batch_len as f64;
                for j in 0..self.n_classes {
                    for k in 0..self.weights_output[j].len() {
                        self.weights_output[j][k] -= scale * (grad_weights_output[j][k] + self.lambda * self.weights_output[j][k]);
                    }
                    self.bias_output[j] -= scale * grad_bias_output[j];
                }

                for h in 0..self.weights_hidden.len() {
                    for i in 0..self.weights_hidden[h].len() {
                        self.weights_hidden[h][i] -= scale * (grad_weights_hidden[h][i] + self.lambda * self.weights_hidden[h][i]);
                    }
                    self.bias_hidden[h] -= scale * grad_bias_hidden[h];
                }
            }

            if epoch % 100 == 0 || epoch == self.epochs - 1 {
                let avg_loss = total_loss / n_samples as f64;
                let accuracy = correct as f64 / n_samples as f64 * 100.0;
                println!("Epoch {}/{} - Loss: {:.4} - Accuracy: {:.2}%", epoch + 1, self.epochs, avg_loss, accuracy);
            }

            if self.weights_output.iter().flatten().any(|w| !w.is_finite()) {
                eprintln!("Poids devenus NaN. Arrêt de l'entraînement.");
                break;
            }
        }
    }

    pub fn predict(&self, x: &[f64]) -> usize {
        let hidden_input: Vec<f64> = self.weights_hidden.iter()
            .zip(self.bias_hidden.iter())
            .map(|(w, &b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.apply_activation(h)).collect();

        let logits: Vec<f64> = self.weights_output.iter()
            .zip(self.bias_output.iter())
            .map(|(w_out, &b)| hidden_output.iter().zip(w_out.iter()).map(|(h, w)| h * w).sum::<f64>() + b)
            .collect();

        if logits.iter().any(|x| !x.is_finite()) {
            eprintln!("[MLPClassifier] Logits non finis : {:?}", logits);
            return 0;
        }

        softmax(&logits)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        let hidden_input: Vec<f64> = self.weights_hidden.iter()
            .zip(self.bias_hidden.iter())
            .map(|(w, &b)| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        let hidden_output: Vec<f64> = hidden_input.iter().map(|&h| self.apply_activation(h)).collect();

        let logits: Vec<f64> = self.weights_output.iter()
            .zip(self.bias_output.iter())
            .map(|(w_out, &b)| hidden_output.iter().zip(w_out.iter()).map(|(h, w)| h * w).sum::<f64>() + b)
            .collect();

        softmax(&logits)
    }


}




    