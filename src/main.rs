
mod linear_model;
use linear_model::linear_gradient_descent::LinearModel;
//mod mlp_model;
mod prepare_dataset;

//use mlp_model::MLP;

fn main() {

    // === 1. Génération du dataset CSV à partir du dossier d’images
    prepare_dataset::generate_dataset_train();
    prepare_dataset::generate_dataset_test();
    
    // Données simples : y = 2x + 1
    let x_data = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
    ];
    let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

    // Initialiser modèle : 1 feature, lr = 0.01, epochs = 1000
    let mut model = LinearModel::new(1, 0.01, 1000);

    // Entraînement (pas d’activation pour la régression)
    model.fit(&x_data, &y_data, None, None);

    // Test de prédiction
    let test_input = vec![5.0];
    let prediction = model.predict(&test_input, None);

    println!("Prédiction pour x = 5.0 : {}", prediction);
}
