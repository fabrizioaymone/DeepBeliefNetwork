//
// Created by whiterose on 12/2/22.
//

#ifndef DBN_DBN_H
#define DBN_DBN_H
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "dataset.h"
#include <iostream>



// The convention adopted follows the one used by Hinton in the paper "A Fast Learning Algorithm for Deep Belief Nets" (2006)
// We will consider the vectors as row vectors
// The name of the layers are: lab <--> top <--> pen --> hid --> vis

class dbn {
public:    // Weight matrices
    MatrixXd hidvis;
    MatrixXd vishid; // not part of generative model
    MatrixXd penhid;
    MatrixXd hidpen; // not part of generative model
    MatrixXd toppen;
    MatrixXd toplab;

    const int BOLTZEPOCHS = 100;

    static double sigmoid(const double);
    void trainboltz(MatrixXd &, std::vector<RowVectorXd>);

    dbn();
    void fit(dataset *);
    int predict(RowVectorXd vis);
    void test(dataset *);



};


#endif //DBN_DBN_H
