use nalgebra::{DMatrix, DVector};

pub enum RBFMode {
    Regression,
    BinaryClassification,
    MultiClassification(usize), // usize = nombre de classes
}

pub struct RBFN {
    pub centers: Vec<Vec<f64>>,     // Chaque point devient un centre
    pub sigma: f64,                 // écart-type de la gaussienne
    pub weights: DMatrix<f64>,     // (n_hidden, n_outputs)
    pub learning_rate: f64,
    pub epochs: usize,
    pub mode: RBFMode,
}

impl RBFN {
    pub fn new(sigma: f64, learning_rate: f64, epochs: usize, mode: RBFMode) -> Self {
        RBFN {
            centers: Vec::new(), // remplis plus tard par x.clone()
            sigma,
            weights: DMatrix::zeros(0, 0), // taille définie plus tard
            learning_rate,
            epochs,
            mode,
        }
    }

    fn gaussian(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-dist_sq / (2.0 * self.sigma.powi(2))).exp()
    }

    fn compute_phi(&self, x: &Vec<Vec<f64>>) -> DMatrix<f64> {
        let n_samples = x.len();
        let n_hidden = self.centers.len();
        let mut data = Vec::with_capacity(n_samples * n_hidden);

        for xi in x {
            for cj in &self.centers {
                data.push(self.gaussian(xi, cj));
            }
        }

        DMatrix::from_row_slice(n_samples, n_hidden, &data)
    }

    /// Fit pour Régression ou Classification Binaire (apprentissage analytique)
    pub fn fit_closed_form(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        self.centers = x.clone(); // RBF Naïf : tous les points deviennent centres
        let n_hidden = self.centers.len();
        let n_outputs = 1;

        let phi = self.compute_phi(x);
        let phi_t = phi.transpose();
        let phi_t_phi = &phi_t * &phi;

        if let Some(inv) = phi_t_phi.try_inverse() {
            let y_matrix = DVector::from_vec(y.clone());
            let weights = inv * (phi_t * y_matrix);
            self.weights = DMatrix::from_columns(&[weights]);
        } else {
            eprintln!("Erreur : matrice non inversible");
        }
    }

    /// Fit pour Classification Multiclasse (descente de gradient)
    pub fn fit_gradient_descent(&mut self, x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) {
        self.centers = x.clone(); // RBF Naïf
        let n_hidden = self.centers.len();
        let n_outputs = match self.mode {
            RBFMode::MultiClassification(k) => k,
            _ => panic!("fit_gradient_descent ne doit être utilisé que pour la classification multiclasse"),
        };

        self.weights = DMatrix::zeros(n_hidden, n_outputs);
        let phi = self.compute_phi(x);

        for _ in 0..self.epochs {
            let prediction = &phi * &self.weights;
            let y_flat: Vec<f64> = y.iter().flat_map(|v| v.iter()).copied().collect();
            let y_matrix = DMatrix::from_row_slice(x.len(), n_outputs, &y_flat);
            let error = &y_matrix - &prediction;
            let gradient = phi.transpose() * error;

            self.weights += self.learning_rate * gradient / (x.len() as f64);
        }
    }

    pub fn predict(&self, x: &Vec<f64>) -> Vec<f64> {
        let phi: Vec<f64> = self.centers.iter().map(|c| self.gaussian(x, c)).collect();
        let phi_vec = DVector::from_vec(phi);
        let result = phi_vec.transpose() * &self.weights;
        result.row(0).iter().copied().collect()
    }

    pub fn predict_label(&self, x: &Vec<f64>) -> f64 {
        match self.mode {
            RBFMode::Regression => self.predict(x)[0],
            RBFMode::BinaryClassification => {
                let out = self.predict(x)[0];
                if out >= 0.0 { 1.0 } else { -1.0 }
            }
            RBFMode::MultiClassification(_) => {
                let output = self.predict(x);
                output.iter()
                      .enumerate()
                      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                      .map(|(i, _)| i as f64)
                      .unwrap_or(-1.0)
            }
        }
    }
}