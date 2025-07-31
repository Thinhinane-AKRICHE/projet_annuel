pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub epochs: usize,
}


impl LinearModel {
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            epochs,
        }
    }

    /// Entraînement (descente de gradient), avec ou sans activation
    pub fn fit(
        &mut self,
        X: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        activation: Option<fn(f64) -> f64>,
        derivative: Option<fn(f64) -> f64>, // dérivée selon output (activation(z))
    ) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                let z = self.predict_raw(xi);

                // activation(z) ou z brut
                let output = match activation {
                    Some(f) => f(z),
                    None => z,
                };

                let error = output - yi;

                // Gradient avec dérivée personnalisée si activation
                let gradient = match derivative {
                    Some(df) => df(output) * error, // dérivée appliquée à output activé
                    None => error,
                };

                for j in 0..self.weights.len() {
                    self.weights[j] -= self.learning_rate * gradient * xi[j];
                }

                self.bias -= self.learning_rate * gradient;
            }
        }
    }

    pub fn predict_raw(&self, x: &Vec<f64>) -> f64 {
        self.weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>() + self.bias
    }

    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> f64 {
        let z = self.predict_raw(x);
        activation.map_or(z, |f| f(z))
    }
}
