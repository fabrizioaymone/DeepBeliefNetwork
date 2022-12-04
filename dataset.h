#ifndef DBN_DATASET_H
#define DBN_DATASET_H



#include <fstream>
#include <stdint.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <unordered_set>

//using row=Eigen::Matrix<double,1,784>;
using namespace Eigen;

class dataset {
public:
    std::vector<RowVectorXd> *train_val_features;
    std::vector<uint8_t> *train_val_labels;


    std::vector<RowVectorXd> *train_features;
    std::vector<RowVectorXd> *valid_features;
    std::vector<RowVectorXd> *test_features;

    std::vector<uint8_t> *train_labels;
    std::vector<uint8_t> *valid_labels;
    std::vector<uint8_t> *test_labels;

    const double TRAIN_SET_PERCENT = 0.90;
    const double VALIDATION_PERCENT = 0.10;

    dataset();
    ~dataset();

    void read_train_val_features(std::string path);
    void read_train_val_labels(std::string path);
    void split_data();
    void read_test_features(std::string path);
    void read_test_labels(std::string path);

    uint32_t convert_to_little_endian(const unsigned char* bytes);

};



#endif //DBN_DATASET_H
