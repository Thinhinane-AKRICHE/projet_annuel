//Méthode des moindres carrés pour le cas de matrices carrées et inversibles 
//la pseudo-inverse formelle

use crate::linear_model::utils::matrix_operations::{transpose, matmul, invert_matrix, matvec_mul};

pub struct LeastSquaresModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LeastSquaresModel {
    /// Entraînement via moindres carrés ordinaires : w = (Xᵗ X)⁻¹ Xᵗ y
    pub fn fit(X: &Vec<Vec<f64>>, y: &Vec<f64>) -> Self {
        let n = X.len();
        let d = X[0].len();

        // Étendre X avec colonne de biais (X_ext : n x (d+1))
        let mut X_ext = vec![vec![1.0; d + 1]; n];
        for i in 0..n {
            X_ext[i][1..].copy_from_slice(&X[i]);
        }

        // Calcul : w = (Xᵗ X)⁻¹ Xᵗ y
        let xt = transpose(&X_ext);
        let xtx = matmul(&xt, &X_ext);
        let xtx_inv = invert_matrix(&xtx).expect("Matrice XᵗX non inversible.");
        let xty = matvec_mul(&xt, y);
        let full_weights = matvec_mul(&xtx_inv, &xty);

        let bias = full_weights[0];
        let weights = full_weights[1..].to_vec();

        Self { weights, bias }
    }

    /// Prédiction brute : w·x + b
    pub fn predict_raw(&self, x: &Vec<f64>) -> f64 {
        self.weights.iter().zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>() + self.bias
    }

    /// Prédiction avec activation facultative (sigmoid, tanh, relu…)
    pub fn predict(&self, x: &Vec<f64>, activation: Option<fn(f64) -> f64>) -> f64 {
        let z = self.predict_raw(x);
        activation.map_or(z, |f| f(z))
    }
}
