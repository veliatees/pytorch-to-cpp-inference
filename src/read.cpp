#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
int main(){
    std::ifstream file("weights/weight1.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> matrix;   // boş başla
    std::string line;
    while(std::getline(file, line)){          // her satır
        std::vector<float> row;               // boş row
        std::stringstream ss(line);
        float val;
        while(ss >> val){                     // satırdaki her sayı
            row.push_back(val);               // row'a ekle
        }
        matrix.push_back(row);                // dolu row'u matrise ekle
    }



    
    file.close();
    return 0;
}
