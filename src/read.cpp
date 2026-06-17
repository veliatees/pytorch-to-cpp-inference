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
    // float func(line){
    //     continue
    // }
        std::vector<std::vector<float>> matrix;   
        std::string line;
        while(std::getline(file, line)){          
            std::vector<float> row;             
            std::stringstream ss(line);
            float val;
            while(ss >> val){                     
                row.push_back(val);               
            }
            matrix.push_back(row);                
        }



    
    file.close();
    return 0;
}
