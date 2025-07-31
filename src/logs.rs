/// Calcule l'erreur quadratique moyenne (Mean Squared Error)
pub fn mse(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    if y_true.len() != y_pred.len() {
        panic!("Les vecteurs y_true et y_pred doivent avoir la même taille");
    }

    let mut sum = 0.0;
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let error = yt - yp;
        sum += error.powi(2);
    }
    sum / y_true.len() as f32
}

/// Calcule la perte logistique (log loss) pour la classification binaire
pub fn log_loss(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    if y_true.len() != y_pred.len() {
        panic!("Les vecteurs y_true et y_pred doivent avoir la même taille");
    }

    let mut sum = 0.0;
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let p = yp.clamp(1e-7, 1.0 - 1e-7); // éviter log(0)
        sum += -yt * p.ln() - (1.0 - yt) * (1.0 - p).ln();
    }
    sum / y_true.len() as f32
}