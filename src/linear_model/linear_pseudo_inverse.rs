//Pour la focntion d'activation on l'ajoute après l'entrainment == pour la prédiction 
//la pseudo-inverse ne peut pas être entraînée avec une fonction d’activation directement : 
//C’est une résolution linéaire analytique, donc on ne peut pas y insérer une activation non linéaire

use nalgebra::{DMatrix, DVector};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct PseudoInverseModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl PseudoInverseModel {
    /// Entraînement via pseudo-inverse (SVD)
    pub fn fit(X: &[Vec<f64>], y: &[f64]) -> Result<Self, Box<dyn Error>> {
        // Vérification des dimensions
        if X.is_empty() || X[0].is_empty() {
            return Err("Matrice d'entrée vide".into());
        }
        
        let n = X.len();
        let d = X[0].len();
        
        // Vérification de la cohérence des dimensions
        if y.len() != n {
            return Err("Dimensions X/y incompatibles".into());
        }

        // Construction de la matrice étendue plus efficacement
        let mut X_ext = DMatrix::<f64>::zeros(n, d + 1);
        for (i, row) in X.iter().enumerate() {
            // Vérification de la cohérence des lignes
            if row.len() != d {
                return Err("Toutes les lignes de X doivent avoir la même longueur".into());
            }
            
            X_ext[(i, 0)] = 1.0; // Colonne de biais
            for (j, &val) in row.iter().enumerate() {
                X_ext[(i, j + 1)] = val;
            }
        }

        let y_vec = DVector::from(y.to_vec());

        // Décomposition SVD avec gestion d'erreur détaillée
        let svd = X_ext.svd(true, true);
        let pinv = svd.pseudo_inverse(1e-8)
            .map_err(|e| format!("Échec du calcul de la pseudo-inverse: {}", e))?;

        // Extraction des poids
        let weights_full = pinv * y_vec;
        let bias = weights_full[0];
        let weights = weights_full.rows(1, d).iter().copied().collect();

        Ok(Self { weights, bias })
    }

    /// Prédiction : w·x + b
    pub fn predict_raw(&self, x: &[f64]) -> Result<f64, Box<dyn Error>> {
        if x.len() != self.weights.len() {
            return Err(format!(
                "Dimension de l'entrée ({}) ne correspond pas aux poids ({})",
                x.len(),
                self.weights.len()
            ).into());
        }
        
        Ok(self.weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>()
            + self.bias)
    }


    /// Prédiction avec activation  (sigmoid, tanh, relu)
    pub fn predict(&self, x: &[f64], activation: Option<fn(f64) -> f64>) -> Result<f64, Box<dyn Error>> {
        let z = self.predict_raw(x)?;
        Ok(activation.map_or(z, |f| f(z)))
}
}
