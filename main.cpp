#include <iostream>
#include "dbn.h"
#include "dataset.h"
#include <Eigen/Dense>

double sigmoid(const double z) {
    return 1.0 / (1.0+exp(-z));
}

int main() {
    dataset * MNIST = new dataset;
    srand(time(0));

    MNIST->read_train_val_features("/home/whiterose/CLionProjects/DBN/dataset/train-images-idx3-ubyte");
    MNIST->read_train_val_labels("/home/whiterose/CLionProjects/DBN/dataset/train-labels-idx1-ubyte");
    MNIST->split_data();
    MNIST->read_test_features("/home/whiterose/CLionProjects/DBN/dataset/t10k-images-idx3-ubyte");
    MNIST->read_test_labels("/home/whiterose/CLionProjects/DBN/dataset/t10k-labels-idx1-ubyte");



    dbn model;

    model.fit(MNIST);

/*

    RowVectorXd v_0(2), h_0(2);
    MatrixXd penhid(2,2);
    v_0 << 1.0, 2.0;
    penhid << 0.0, 1.0, 1.0, 0.0;
    h_0 = v_0 * penhid.transpose();
    std::cout << h_0<< std::endl;
    h_0 = h_0.unaryExpr(std::ref(sigmoid));
    std::cout << h_0<< std::endl;
    RowVectorXd rand_vec;
    rand_vec = (RowVectorXd::Random(2) + RowVectorXd::Constant(2,1.0))/2.0 ;
    std::cout << rand_vec << std::endl;
    std::cout << (h_0.array() > rand_vec.array())<< std::endl;


*/

}


