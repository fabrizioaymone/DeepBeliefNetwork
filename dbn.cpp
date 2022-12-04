#include "dbn.h"

dbn::dbn(){
    hidvis = MatrixXd::Random(500,784);
    vishid = hidvis.transpose();
    penhid = MatrixXd::Random(500,500);
    hidpen = penhid.transpose();
    toppen = MatrixXd::Random(200,500);
    toplab = MatrixXd::Random(200,10);
}

void dbn::fit(dataset * d) {

    // Greedy initial learning

    //train first Boltzmann machine

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for 1st BM" << std::endl;
        for (int i = 9;i < 10; i+=10) {
            //std::cout << hidvis(0,0) << std::endl;
            std::vector<RowVectorXd> features_vec = {d->train_features->begin(), d->train_features->begin()+i};
            trainboltz(hidvis, features_vec);
        }
    }

    vishid = hidvis.transpose();

    //train second Boltzmann machine

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for 2nd BM" << std::endl;
        for (int i = 9;i < 10; i+=10) {
            std::vector<RowVectorXd> features_vec = {d->train_features->begin(), d->train_features->begin()+i};
            for(RowVectorXd &i : features_vec)
                i = i * vishid;
            trainboltz(penhid, features_vec);
            //RowVectorXd vis = d->train_features->at(i) * vishid;
            //trainboltz(penhid, vis);
        }
    }
    hidpen = penhid.transpose();

    //top_layer

    // vis layer = 10 one-hot-encoded layer + 500 pen layer
    // W_Concat = toplabel + toppen

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for Top Associative Memory" << std::endl;
        for (int i = 9; i < 10; i+=10) {
            std::vector<RowVectorXd> features_vec;
            for(int j=i-9; j<=i; j++){
            RowVectorXd label = RowVectorXd::Zero(10);
            label[d->train_labels->at(j)] = 1;
            RowVectorXd pen = (d->train_features->at(j) * vishid) * hidpen;
            RowVectorXd vis(510);
            vis << label, pen;
            features_vec.push_back(vis);
        }
            MatrixXd W_concat(200, 510);
            W_concat << toplab, toppen;
            trainboltz(W_concat, features_vec);

            toplab = W_concat.leftCols(10);
            toppen = W_concat.rightCols(500);
        }
    }

    //Save weights

    std::cout << "Beginning Validation"<<std::endl;
    int correct = 0;
    for(int i = 0; i < 10; i++){
        RowVectorXd vis = d->train_features->at(i);
        int label = d->train_labels->at(i);
        int predicted = predict(vis);
        std::cout << label << " " << predicted << std::endl;
        if(label == predicted)
            correct++;
    }

    std::cout << "The total number of correct predictions is " << correct << " with an accuracy of the " << (double) correct/(10)*100 <<"%"<<std::endl;


    //Validation
 /*   std::cout << "Beginning Validation"<<std::endl;
    int correct = 0;
    for(int i = 0; i < d->valid_features->size(); i++){
        RowVectorXd vis = d->valid_features->at(i);
        int label = d->valid_labels->at(i);
        int predicted = predict(vis);
        std::cout << label << " " << predict(vis) << std::endl;
        if(label == predicted)
            correct++;
    }

    std::cout << "The total number of correct predictions is " << correct << " with an accuracy of the " << (double) correct/(d->valid_features->size())*100 <<"%"<<std::endl;
    */
}

int dbn::predict(Eigen::RowVectorXd vis) {
    RowVectorXd hid, pen, top, label, rand_vec;

    std::cout << vis << std::endl;

    hid = vis * vishid;
    rand_vec = (RowVectorXd::Random(hid.cols()) + RowVectorXd::Constant(hid.cols(), 1.0)) / 2.0;
    hid = (hid.array() > rand_vec.array()).cast<double>();

    pen = hid * hidpen;
    rand_vec = (RowVectorXd::Random(pen.cols()) + RowVectorXd::Constant(pen.cols(), 1.0)) / 2.0;
    pen = (pen.array() > rand_vec.array()).cast<double>();

    top = pen * toppen.transpose();
    rand_vec = (RowVectorXd::Random(top.cols()) + RowVectorXd::Constant(top.cols(), 1.0)) / 2.0;
    top = (top.array() > rand_vec.array()).cast<double>();

    label = top * toplab;

    int idx = 0;
    double max = label(0,0);
    for(int i = 0; i < label.cols(); i++){
        if(label(0,i)>max){
            max = label(0,i);
            idx=i;
        }
    }
    std::cout << "label prediction are " << label << std::endl;

    return idx;
}

void dbn::trainboltz(MatrixXd & W, std::vector<RowVectorXd> vis_vec){
    RowVectorXd v_0, v_1, v_0_avg, v_1_avg;
    RowVectorXd h_0, h_1, h_0_avg,  h_1_avg;
    RowVectorXd rand_vec;

    v_0_avg = RowVectorXd::Zero(W.cols());
    h_0_avg = RowVectorXd::Zero(W.rows());
    v_1_avg = RowVectorXd::Zero(W.cols());
    h_1_avg = RowVectorXd::Zero(W.rows());

    for(int i = 0; i < vis_vec.size(); i++){
    //positive phase
    v_0 = vis_vec.at(i);
    h_0 = v_0 * W.transpose();
    h_0 = h_0.unaryExpr(std::ref(sigmoid));
    rand_vec = (RowVectorXd::Random(h_0.cols()) + RowVectorXd::Constant(h_0.cols(), 1.0)) / 2.0;
    h_0 = (h_0.array() > rand_vec.array()).cast<double>();

    //negative phase
    v_1 = h_0 * W;
    v_1 = v_1.unaryExpr(std::ref(sigmoid));
    rand_vec = (RowVectorXd::Random(v_1.cols()) + RowVectorXd::Constant(v_1.cols(), 1.0)) / 2.0;
    v_1 = (v_1.array() > rand_vec.array()).cast<double>();

    h_1 = v_1 * W.transpose();
    h_1 = h_1.unaryExpr(std::ref(sigmoid));
    rand_vec = (RowVectorXd::Random(h_1.cols()) + RowVectorXd::Constant(h_1.cols(), 1.0)) / 2.0;
    h_1 = (h_1.array() > rand_vec.array()).cast<double>();

    v_0_avg += v_0;
    h_0_avg += h_0;
    v_1_avg += v_1;
    h_1_avg += h_1;
}
    v_0_avg/= vis_vec.size();
    h_0_avg/= vis_vec.size();
    v_1_avg/= vis_vec.size();
    h_1_avg/= vis_vec.size();


    // weight update
    W += h_0_avg.transpose() * v_0_avg - h_1_avg.transpose() * v_1_avg;
}


double dbn::sigmoid(const double z) {
    return 1.0 / (1.0+exp(-z));
}
