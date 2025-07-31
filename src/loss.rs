pub fn mse(pred: &[f32], target: &[f32]) -> Option<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return None;
    }
    let sum: f32 = pred.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    Some(sum / pred.len() as f32)
}