pub struct PerceptronModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl PerceptronModel {
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            epochs,
        }
    }

    pub fn fit(&mut self, X: &[Vec<f64>], y: &[f64]) {
        for _ in 0..self.epochs {
            for (xi, &yi) in X.iter().zip(y.iter()) {
                let z = self.weights.iter().zip(xi.iter()).map(|(w, x)| w * x).sum::<f64>() + self.bias;
                let y_pred = if z >= 0.0 { 1.0 } else { -1.0 };
                let error = yi - y_pred;

                for (w, &xij) in self.weights.iter_mut().zip(xi.iter()) {
                    *w += self.learning_rate * error * xij;
                }
                self.bias += self.learning_rate * error;
            }
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let z = self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias;
        if z >= 0.0 { 1.0 } else { -1.0 }
    }
}
