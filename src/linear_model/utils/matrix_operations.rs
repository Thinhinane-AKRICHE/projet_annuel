/// Transpose une matrice m[n][d] -> m[d][n]
pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

/// Multiplication de matrices : A[n][d] * B[d][k] = C[n][k]
pub fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();

    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Multiplication matrice-vecteur : A[n][d] * x[d] = y[n]
pub fn matvec_mul(a: &Vec<Vec<f64>>, x: &Vec<f64>) -> Vec<f64> {
    let rows = a.len();
    let mut result = vec![0.0; rows];

    for i in 0..rows {
        for j in 0..x.len() {
            result[i] += a[i][j] * x[j];
        }
    }
    result
}

/// Inversion d'une matrice carrée par la méthode de Gauss-Jordan
pub fn invert_matrix(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();

    // Vérification que la matrice est carrée
    if matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    // Créer la matrice augmentée [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Méthode de Gauss-Jordan
    for i in 0..n {
        // Pivot non nul
        let pivot = aug[i][i];
        if pivot.abs() < 1e-10 {
            return None; // Matrice non inversible
        }

        // Normaliser la ligne
        for j in 0..2 * n {
            aug[i][j] /= pivot;
        }

        // Éliminer les autres lignes
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extraire la partie droite : l'inverse
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }

    Some(inverse)
}
