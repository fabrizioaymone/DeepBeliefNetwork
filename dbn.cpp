#include "dbn.h"
#define DEBUG BATCHSIZE

dbn::dbn(){
    hidvis = MatrixXd::Random(500,784);
    vishid = hidvis.transpose();
    penhid = MatrixXd::Random(500,500);
    hidpen = penhid.transpose();
    toppen = MatrixXd::Random(200,500);
    toplab = MatrixXd::Random(200,10);
}

void dbn::fit(dataset * d) {
    //FIRST LAYER

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for 1st BM" << std::endl;
        for (int i = BATCHSIZE;i < DEBUG+1; i+=BATCHSIZE) {
            //std::cout << hidvis(0,0) << std::endl;
            std::vector<RowVectorXd> features_vec = {d->train_features->begin()+i-BATCHSIZE, d->train_features->begin()+i};
            trainboltz(hidvis, features_vec);
        }
    }

    vishid = hidvis.transpose();

    //SECOND LAYER

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for 2nd BM" << std::endl;
        for (int i = BATCHSIZE;i < DEBUG+1; i+=BATCHSIZE) {
            std::vector<RowVectorXd> features_vec = {d->train_features->begin()+i-BATCHSIZE, d->train_features->begin()+i};
            for(RowVectorXd &i : features_vec)
                i = i * vishid;
            trainboltz(penhid, features_vec);
            //RowVectorXd vis = d->train_features->at(i) * vishid;
            //trainboltz(penhid, vis);
        }
    }
    hidpen = penhid.transpose();

    //ASSOCIATIVE MEMORY

    for(int j = 0; j < BOLTZEPOCHS; j++) {
        std::cout << "Training Epoch: " << j+1 <<" for Top Associative Memory" << std::endl;
        for (int i = BATCHSIZE-1;i < DEBUG; i+=BATCHSIZE) {
            std::vector<RowVectorXd> pen_vec = {d->train_features->begin()+i-BATCHSIZE+1, d->train_features->begin()+i};
            for(RowVectorXd &k : pen_vec)
                k = (k * vishid) * hidpen;
            std::vector<RowVectorXd> lab_vec;
            for(int k=i-BATCHSIZE+1; k<=i; k++){
                RowVectorXd label = RowVectorXd::Zero(10);
                label(0, d->train_labels->at(k)) = 1;
                lab_vec.push_back(label);
            }
            trainmemboltz(toppen, toplab, pen_vec, lab_vec);

        }
    }


    std::cout << "Beginning Validation"<<std::endl;
    int correct = 0;
    for(int i = 0;i < DEBUG; i++){
        RowVectorXd vis = d->train_features->at(i);
        int label = d->train_labels->at(i);
        int predicted = predict(vis);
        std::cout << label << " " << predicted << std::endl;
        if(label == predicted)
            correct++;
    }

    std::cout << "The total number of correct predictions is " << correct << " with an accuracy of the " << (double) correct/(DEBUG)*100 <<"%"<<std::endl;


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

    hid = vis * vishid;
    hid = hid.unaryExpr(std::ref(sigmoid));
    //rand_vec = (RowVectorXd::Random(hid.cols()) + RowVectorXd::Constant(hid.cols(), 1.0)) / 2.0;
    //hid = (hid.array() > rand_vec.array()).cast<double>();

    pen = hid * hidpen;
    pen = pen.unaryExpr(std::ref(sigmoid));
    //rand_vec = (RowVectorXd::Random(pen.cols()) + RowVectorXd::Constant(pen.cols(), 1.0)) / 2.0;
    //pen = (pen.array() > rand_vec.array()).cast<double>();

   std::cout<< "pen layer" << pen << std::endl;
   label = RowVectorXd::Constant(10, 0.1);

   //Alternating Gibbs
   for(int i=0; i<2; i++) {
       top = pen * toppen.transpose() + label * toplab.transpose();
       top = top.unaryExpr(std::ref(sigmoid));
       //rand_vec = (RowVectorXd::Random(top.cols()) + RowVectorXd::Constant(top.cols(), 1.0)) / 2.0;
       //top = (top.array() > rand_vec.array()).cast<double>();

       label = top * toplab;
       label = softmax(label);
       std::cout << "label prediction are " << label << std::endl;
       //rand_vec = (RowVectorXd::Random(label.cols()) + RowVectorXd::Constant(label.cols(), 1.0)) / 2.0;
       //label = (label.array() > rand_vec.array()).cast<double>();

       pen = top * toppen;
       pen = pen.unaryExpr(std::ref(sigmoid));
       rand_vec = (RowVectorXd::Random(pen.cols()) + RowVectorXd::Constant(pen.cols(), 1.0)) / 2.0;
       pen = (pen.array() > rand_vec.array()).cast<double>();
   }

    //rand_vec = (RowVectorXd::Random(top.cols()) + RowVectorXd::Constant(top.cols(), 1.0)) / 2.0;
    //top = (top.array() > rand_vec.array()).cast<double>();

    //std::cout << "top layer is " << top << std::endl:


    int idx = 0;
    double max = label(0,0);
    for(int i = 0; i < label.cols(); i++){
        if(label(0,i)>max){
            max = label(0,i);
            idx=i;
        }
    }


    return idx;
}

void dbn::trainboltz(MatrixXd & W, std::vector<RowVectorXd> vis_vec){
    RowVectorXd v_0, v_1;
    RowVectorXd h_0, h_1;
    RowVectorXd rand_vec;
    MatrixXd delta_W;

    delta_W = MatrixXd::Zero(W.rows(), W.cols());

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


    delta_W += h_0.transpose() * v_0 - h_1.transpose() * v_1;

}
    delta_W /= vis_vec.size();


    // weight update
    W += delta_W;
   // std::cout  << "Weight update is " << (h_0_avg.transpose() * v_0_avg - h_1_avg.transpose() * v_1_avg).array() << std::endl;
}

void dbn::trainmemboltz(MatrixXd & toppen, MatrixXd & toplab, std::vector<RowVectorXd> pen_vec, std::vector<RowVectorXd> lab_vec){
    RowVectorXd pen_0, lab_0, pen_1, lab_1;
    RowVectorXd top_0, top_1;
    MatrixXd delta_toppen, delta_toplab;
    RowVectorXd pen_rand_vec, lab_rand_vec, top_rand_vec;

    const int batch_size = pen_vec.size();
    delta_toppen = MatrixXd::Zero(toppen.rows(), toppen.cols());
    delta_toplab = MatrixXd::Zero(toplab.rows(), toplab.cols());

    for(int i = 0; i < pen_vec.size(); i++){
        //positive phase
        pen_0 = pen_vec.at(i);
        lab_0 = lab_vec.at(i);
        top_0 = (pen_0 * toppen.transpose()) + (lab_0 * toplab.transpose());
        top_0 = top_0.unaryExpr(std::ref(sigmoid));
        top_rand_vec = (RowVectorXd::Random(top_0.cols()) + RowVectorXd::Constant(top_0.cols(), 1.0)) / 2.0;
        top_0 = (top_0.array() > top_rand_vec.array()).cast<double>();

        //negative phase
        pen_1 = top_0 * toppen;
        lab_1 = top_0 * toplab;
        pen_1 = pen_1.unaryExpr(std::ref(sigmoid));
        lab_1 = softmax(lab_1);
        pen_rand_vec = (RowVectorXd::Random(pen_1.cols()) + RowVectorXd::Constant(pen_1.cols(), 1.0)) / 2.0;
        lab_rand_vec = (RowVectorXd::Random(lab_1.cols()) + RowVectorXd::Constant(lab_1.cols(), 1.0)) / 2.0;
        pen_1 = (pen_1.array() > pen_rand_vec.array()).cast<double>();
        lab_1 = (lab_1.array() > lab_rand_vec.array()).cast<double>();

        top_1 = (pen_1 * toppen.transpose()) +(lab_1 * toplab.transpose());
        top_1 = top_1.unaryExpr(std::ref(sigmoid));
        top_rand_vec = (RowVectorXd::Random(top_1.cols()) + RowVectorXd::Constant(top_1.cols(), 1.0)) / 2.0;
        top_1 = (top_1.array() > top_rand_vec.array()).cast<double>();

        delta_toppen += top_0.transpose() * pen_0 - top_1.transpose() * pen_1;
        delta_toplab += top_0.transpose() * lab_0 - top_1.transpose() * lab_1;

    }
    delta_toppen/=batch_size;
    delta_toplab/=batch_size;

    // weight update
    toppen += delta_toppen;
    toplab += delta_toplab;
}


RowVectorXd dbn::softmax(RowVectorXd label){
    double sum=0;
    for(int i=0; i < label.cols(); i++){
        sum+=exp(label(0,i));
    }
    for(int i=0; i<label.cols(); i++){
        label(0,i) = exp(label(0,i))/sum;
    }
    return label;
}

double dbn::sigmoid(const double z) {
    return 1.0 / (1.0+exp(-z));
}
