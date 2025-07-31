use crate::linear_model::utils::activations::softmax;
use rand::Rng;

pub struct SoftmaxModel {
    pub weights: Vec<Vec<f64>>,  // [n_classes][n_features]
    pub biases: Vec<f64>,        // [n_classes]
    pub learning_rate: f64,
    pub epochs: usize,
    pub n_classes: usize,
    pub lambda: f64,             // Coefficient de régularisation L2
    // === Ajout pour le suivi ===
    pub train_losses: Vec<f64>,
    pub test_losses: Vec<f64>,
    pub train_accuracies: Vec<f64>,
    pub test_accuracies: Vec<f64>,
}

impl SoftmaxModel {
    pub fn new(
        n_features: usize, 
        n_classes: usize, 
        learning_rate: f64, 
        epochs: usize, 
        lambda: f64
        ) -> Self {
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
            lambda,
            train_losses: Vec::with_capacity(epochs),
            test_losses: Vec::with_capacity(epochs),
            train_accuracies: Vec::with_capacity(epochs),
            test_accuracies: Vec::with_capacity(epochs),
        }
    }

    pub fn fit(
    &mut self,
    X_train: &Vec<Vec<f64>>,
    y_train: &Vec<usize>,
    X_test: Option<&Vec<Vec<f64>>>,
    y_test: Option<&Vec<usize>>,
) {
    let n = X_train.len();
    let d = X_train[0].len();

    assert_eq!(y_train.len(), n, "Longueur de y différente de X");
    assert_eq!(self.weights.len(), self.n_classes, "Poids mal initialisés");
    assert_eq!(self.biases.len(), self.n_classes);

    for row in X_train {
        assert_eq!(row.len(), d, "Incohérence dans les dimensions de X");
    }

    for w in &self.weights {
        assert_eq!(w.len(), d, "Incohérence dans les dimensions des poids");
    }

    println!("Démarrage de l'entraînement Softmax : {} epochs", self.epochs);

    for epoch in 0..self.epochs {
        let mut total_loss = 0.0;

        for i in 0..n {
            let xi = &X_train[i];
            let yi = y_train[i];
            assert!(yi < self.n_classes, "Classe {yi} hors des bornes !");

            // === 1. Calcul des scores ===
            let mut scores = vec![0.0; self.n_classes];
            for k in 0..self.n_classes {
                scores[k] = self.weights[k]
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.biases[k];
            }

            // === 2. Softmax ===
            let probs = softmax(&scores);
            if probs.iter().any(|p| !p.is_finite()) {
                panic!("Probas invalides : {:?}", probs);
            }

            // === 3. Cross-entropy loss ===
            let epsilon = 1e-15;
            let clipped_prob = probs[yi].max(epsilon);
            total_loss -= clipped_prob.ln();

            // === 4. Mise à jour des poids et biais ===
            for k in 0..self.n_classes {
                let error = probs[k] - if k == yi { 1.0 } else { 0.0 };

                for j in 0..d {
                    self.weights[k][j] -= self.learning_rate
                        * (error * xi[j] + self.lambda * self.weights[k][j]);
                }

                self.biases[k] -= self.learning_rate * error;
            }
        }

        // === 5. Statistiques de l'époque ===
        let avg_loss = total_loss / n as f64;
        self.train_losses.push(avg_loss);
        self.train_accuracies.push(self.accuracy(X_train, y_train));

        if let (Some(Xt), Some(yt)) = (X_test, y_test) {
            self.test_losses.push(self.compute_loss(Xt, yt));
            self.test_accuracies.push(self.accuracy(Xt, yt));
        }

        if epoch % 100 == 0 || epoch == self.epochs - 1 {
            println!(
                "Epoch {}/{} — Train Loss: {:.4} — Train Acc: {:.2}% — Test Acc: {:.2}%",
                epoch + 1,
                self.epochs,
                avg_loss,
                self.train_accuracies.last().unwrap_or(&0.0) * 100.0,
                self.test_accuracies.last().unwrap_or(&0.0) * 100.0
            );
        }
    }

    println!("Entraînement terminé !");
}


    pub fn compute_loss(&self, X: &Vec<Vec<f64>>, y: &Vec<usize>) -> f64 {
        let mut loss = 0.0;
        let epsilon = 1e-15;

        for (xi, &yi) in X.iter().zip(y.iter()) {
            let probs = self.predict_proba(xi);
            let clipped = probs[yi].max(epsilon);
            loss -= clipped.ln();
        }

        loss / X.len() as f64
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
        let probs = self.predict_proba(x);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn predict_proba(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut scores = vec![0.0; self.n_classes];
        for k in 0..self.n_classes {
            scores[k] = self.weights[k]
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum::<f64>()
                + self.biases[k];
        }

        softmax(&scores)
    }
}
