#include "dataset.h"

dataset::dataset(){
    train_val_features = new std::vector<RowVectorXd>;
    train_features = new std::vector<RowVectorXd>;
    valid_features = new std::vector<RowVectorXd>;
    test_features =  new std::vector<RowVectorXd>;
    train_val_labels = new std::vector<uint8_t>;
    train_labels = new std::vector<uint8_t>;
    valid_labels = new std::vector<uint8_t>;
    test_labels = new std::vector<uint8_t>;
}

dataset::~dataset(){
}

void dataset::read_train_val_features(std::string path){
    uint32_t header[4];     // |MAGIC|NUM IMAGES| ROWSIZE | COLSIZE
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f){
        for(int i = 0; i < 4; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Input file header.\n");
        int  image_size = header[2]*header[3];
        for(int i = 0; i < header[1]; i++){
            RowVectorXd tmp(784);
            uint8_t element[1];
            double ok;
            for(int j=0; j < image_size; j++){
                if(fread(element, sizeof(element[0]), 1, f)){
                    tmp(0,j)= element[0]/255.0;
                }else{
                    printf("Error Reading from File.\n");
                    exit(1);
                }
            }
            train_val_features->push_back(tmp);
        }
        printf("Successfully read and stored %lu train_val feature vectors.\n", train_val_features->size());
    }else{
        printf("Could not open train_val file");
        exit(1);
    }
}
void dataset::read_test_features(std::string path){
    uint32_t header[4];     // |MAGIC|NUM IMAGES| ROWSIZE | COLSIZE
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f){
        for(int i = 0; i < 4; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Input file header.\n");
        int  image_size = header[2]*header[3];
        for(int i = 0; i < header[1]; i++){
            RowVectorXd tmp(784);
            uint8_t element[1];
            for(int j=0; j < image_size; j++){
                if(fread(element, sizeof(element[0]), 1, f)){
                    tmp(0,j) = element[0]/255.0;
                }else{
                    printf("Error Reading from File.\n");
                    exit(1);
                }
            }
            test_features->push_back(tmp);
        }
        printf("Successfully read and stored %lu test feature vectors.\n", test_features->size());
    }else{
        printf("Could not open file");
        exit(1);
    }
}
void dataset::read_train_val_labels(std::string path){
    uint32_t header[2];     // |MAGIC|NUM LABELS
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if(f){
        for(int i = 0; i < 2; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting label file header.\n");
        for(int i = 0; i < header[1]; i++){
            uint8_t element[1];
            if(fread(element, sizeof(element[0]), 1, f)){
                train_val_labels->push_back(element[0]);
            }else{
                printf("Erro Reading from File.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored %lu train_val labels.\n", train_val_labels->size());
    }else{
        printf("Could not open file");
        exit(1);
    }
}
void dataset::read_test_labels(std::string path) {
    uint32_t header[2];     // |MAGIC|NUM LABELS
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if(f){
        for(int i = 0; i < 2; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting label file header.\n");
        for(int i = 0; i < header[1]; i++){
            uint8_t element[1];
            if(fread(element, sizeof(element[0]), 1, f)){
                test_labels->push_back(element[0]);
            }else{
                printf("Erro Reading from File.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored %lu test labels.\n", train_val_labels->size());
    }else{
        printf("Could not open file");
        exit(1);
    }
}
void dataset::split_data(){
    std::unordered_set<int> used_indexes;
    int train_size = train_val_features->size() * TRAIN_SET_PERCENT;
    int validation_size = train_val_features->size() * VALIDATION_PERCENT;

    // training data

    int count = 0;
    while(count < train_size){
        int rand_index = rand() % (train_val_features->size());
        if(used_indexes.find(rand_index) == used_indexes.end()){
            train_features->push_back(train_val_features->at(rand_index));
            train_labels->push_back(train_val_labels->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    // validation data

    for(int rand_index=0; rand_index < train_val_features->size(); rand_index++)
        if(used_indexes.find(rand_index) == used_indexes.end()){
            valid_features->push_back(train_val_features->at(rand_index));
            valid_labels->push_back(train_val_labels->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }


    printf("Training Data Size: %lu.\n", train_features->size());
    printf("Validation Data Size: %lu.\n", valid_features->size());

}

uint32_t dataset::convert_to_little_endian(const unsigned char* bytes){

    uint32_t ret = (uint32_t) ((bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3]);
    return ret;
}
