mod linear_model;
mod mlp_model;

use linear_model::{LinearModel, tanh, tanh_derivative};
use mlp_model::MLP;

fn main() {
    // Exemple de données
    let X = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![1.5, 1.5]]; //x une matrice 3*2 (nbr de features = 2) et on a trois exemples / échantilloons
    let y_regression = vec![3.0, 3.0, 3.0]; //y_regression = un vecteur 3*1
    let y_classification = vec![1.0, -1.0, 1.0]; //y_classification = un vecteur 3*1 (il prend les valuers 1 et -1)

    
    // --- Linear Model pour regression
    let mut lm = LinearModel::new(2, 0.01, 1000);
    lm.fit(&X, &y_regression, None, None); // pas d'activation
    println!("Linear regression prediction: {}", lm.predict(&X[0], None)); //on teste sur le premier exemple : predict [1.0, 2.0] -> 3.0
    
    // --- Linear Model pour classification (avec tanh)
    let mut lm_classif = LinearModel::new(2, 0.9, 1000);
    lm_classif.fit(&X, &y_classification, Some(tanh), Some(tanh_derivative));
    println!("Linear classification prediction: {}", lm_classif.predict(&X[1], Some(tanh)));
    /* 
    // --- MLP Model
    let mut mlp = MLP::new(2, 4, 0.01, 1000);
    mlp.fit(&X, &y_classification);
    println!("MLP prediction: {}", mlp.predict(&X[0]));
     */
}
