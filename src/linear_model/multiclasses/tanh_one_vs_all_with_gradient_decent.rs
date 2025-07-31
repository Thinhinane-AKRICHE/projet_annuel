use rand::Rng;
use crate::linear_model::utils::activations::tanh;

#[repr(C)]
pub struct OneVsAllTanhMSEModel {
    pub weights: Vec<Vec<f64>>,     // [n_classes][n_features]
    pub biases: Vec<f64>,           // [n_classes]
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
    pub losses: Vec<f64>,           // train loss par epoch
    pub test_losses: Vec<f64>,      // test loss par epoch
    pub train_accuracies: Vec<f64>, // train accuracy par epoch
    pub test_accuracies: Vec<f64>,  // test accuracy par epoch
}

impl OneVsAllTanhMSEModel {
    pub fn new(n_features: usize, n_classes: usize, learning_rate: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..n_classes)
            .map(|_| (0..n_features).map(|_| rng.gen_range(-0.01..0.01)).collect())
            .collect();

        Self {
            weights,
            biases: vec![0.0; n_classes],
            learning_rate,
            epochs,
            n_classes,
            losses: Vec::with_capacity(epochs),
            test_losses: Vec::with_capacity(epochs),
            train_accuracies: Vec::with_capacity(epochs),
            test_accuracies: Vec::with_capacity(epochs),
        }
    }

    pub fn fit(&mut self,
               X_train: &Vec<Vec<f64>>,
               y_train: &Vec<usize>,
               X_test: Option<&Vec<Vec<f64>>>,
               y_test: Option<&Vec<usize>>) {
        let n = X_train.len();
        let d = X_train[0].len();

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            for i in 0..n {
                let xi = &X_train[i];
                let yi = y_train[i];

                let mut outputs = vec![0.0; self.n_classes];
                for k in 0..self.n_classes {
                    let z = self.weights[k]
                        .iter()
                        .zip(xi.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>()
                        + self.biases[k];
                    outputs[k] = tanh(z);
                }

                let targets: Vec<f64> = (0..self.n_classes)   //rée un intervalle de toutes les classes possibles exp : 0->3 : [0,1,2]
                    .map(|k| if k == yi { 1.0 } else { 0.0 }) //pour chaque classe si vrai étiquette -> 1.0 sinon 0.0
                    .collect();

                for k in 0..self.n_classes {
                    let error = outputs[k] - targets[k];
                    total_loss += 0.5 * error * error;  

                    let grad = error * (1.0 - outputs[k].powi(2)); // dérivée tanh

                    for j in 0..d {
                        self.weights[k][j] -= self.learning_rate * grad * xi[j];
                    }

                    self.biases[k] -= self.learning_rate * grad;
                }
            }

            // === Train loss + accuracy ===
            let avg_loss = total_loss / n as f64;
            self.losses.push(avg_loss);

            let train_acc = self.accuracy(X_train, y_train);
            self.train_accuracies.push(train_acc);

            // === Test loss + accuracy ===
            if let (Some(Xt), Some(yt)) = (X_test, y_test) {
                let test_loss = self.compute_loss(Xt, yt);
                self.test_losses.push(test_loss);

                let test_acc = self.accuracy(Xt, yt);
                self.test_accuracies.push(test_acc);
            }

            if epoch % 10 == 0 || epoch == self.epochs - 1 {
                println!(
                    "Epoch {}/{} — Train loss: {:.6} — Train acc: {:.2}% — Test acc: {:.2}%",
                    epoch + 1,
                    self.epochs,
                    avg_loss,
                    train_acc * 100.0,
                    self.test_accuracies.last().unwrap_or(&0.0) * 100.0
                );
            }
        }

        println!("Entraînement terminé !");
    }

    pub fn compute_loss(&self, X: &Vec<Vec<f64>>, y: &Vec<usize>) -> f64 {
        let mut total_loss = 0.0;

        for (xi, &yi) in X.iter().zip(y.iter()) {
            let outputs: Vec<f64> = (0..self.n_classes)
                .map(|k| {
                    let z = self.weights[k]
                        .iter()
                        .zip(xi.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>()
                        + self.biases[k];
                    tanh(z)
                })
                .collect();

            let targets: Vec<f64> = (0..self.n_classes)
                .map(|k| if k == yi { 1.0 } else { 0.0 })
                .collect();

            for k in 0..self.n_classes {
                let error = outputs[k] - targets[k];
                total_loss += 0.5 * error * error;
            }
        }

        total_loss / X.len() as f64
    }

    pub fn accuracy(&self, X: &Vec<Vec<f64>>, y: &Vec<usize>) -> f64 {
        let mut correct = 0;
        for (xi, &yi) in X.iter().zip(y.iter()) {
            let pred = self.predict(xi);
            if pred == yi {
                correct += 1;
            }
        }
        correct as f64 / X.len() as f64
    }

    pub fn predict(&self, x: &Vec<f64>) -> usize {
        let mut scores = vec![0.0; self.n_classes];
        for k in 0..self.n_classes {
            let z = self.weights[k]
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum::<f64>()
                + self.biases[k];
            scores[k] = tanh(z);
        }

        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn predict_scores(&self, x: &Vec<f64>) -> Vec<f64> {
        (0..self.n_classes)
            .map(|k| {
                let z = self.weights[k]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| w * xi)
                    .sum::<f64>()
                    + self.biases[k];
                tanh(z)
            })
            .collect()
    }
}
